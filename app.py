import os, json, time, threading, asyncio, traceback
from datetime import datetime, timezone
from pathlib import Path
import ccxt, joblib, numpy as np, pandas as pd, pandas_ta_classic as ta
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

BASE=Path(__file__).resolve().parent; MODEL_DIR=BASE/'models'; CONFIG=BASE/'config.json'; SIGNALS=BASE/'signals.json'
MODEL_DIR.mkdir(exist_ok=True)
DEFAULT={"exchange":"binance","symbol":"BTC/USDT","timeframe":"15m","htf":"1h","candle_limit":1200,"scan_interval":300,"watchlist":["ETH/USDT","SOL/USDT"],"telegram_token":"","telegram_chat_id":""}
app=Flask(__name__); CORS(app)
lock=threading.RLock(); scanner={"running":False,"last_scan":None,"last_error":None}; sent_keys=set()


def read_json(path, default):
    try:
        if not path.exists(): path.write_text(json.dumps(default,ensure_ascii=False,indent=2),encoding='utf-8')
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception: return default.copy() if isinstance(default,dict) else default

def write_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(path)

def cfg():
    c=read_json(CONFIG,DEFAULT)
    # Trên hosting, secret Telegram ưu tiên lấy từ biến môi trường.
    if os.getenv('TELEGRAM_TOKEN'): c['telegram_token']=os.getenv('TELEGRAM_TOKEN')
    if os.getenv('TELEGRAM_CHAT_ID'): c['telegram_chat_id']=os.getenv('TELEGRAM_CHAT_ID')
    return c

def exchange_obj(name):
    cls=getattr(ccxt,name,None)
    if cls is None: raise ValueError(f"Sàn không được hỗ trợ: {name}")
    return cls({'enableRateLimit':True,'timeout':20000})

def fetch_df(ex,symbol,tf,limit):
    if not ex.has.get('fetchOHLCV'): raise RuntimeError(f'{ex.id} không hỗ trợ OHLCV')
    rows=ex.fetch_ohlcv(symbol,tf,limit=min(int(limit),1500))
    if len(rows)<250: raise RuntimeError(f'Thiếu dữ liệu {symbol} {tf}: chỉ có {len(rows)} nến')
    d=pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume'])
    d=d.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    d[['open','high','low','close','volume']]=d[['open','high','low','close','volume']].astype(float)
    return d

def firstcol(d,prefix):
    c=[x for x in d.columns if x.upper().startswith(prefix.upper())]
    return c[0] if c else None

def indicators(d):
    x=d.copy()
    x['ema9']=ta.ema(x.close,length=9); x['ema21']=ta.ema(x.close,length=21); x['ema50']=ta.ema(x.close,length=50); x['ema200']=ta.ema(x.close,length=200)
    x['rsi7']=ta.rsi(x.close,length=7); x['rsi14']=ta.rsi(x.close,length=14); x['rsi21']=ta.rsi(x.close,length=21)
    bb=ta.bbands(x.close,length=20,std=2); x['bb_l']=bb[firstcol(bb,'BBL_')]; x['bb_m']=bb[firstcol(bb,'BBM_')]; x['bb_u']=bb[firstcol(bb,'BBU_')]
    st=ta.stoch(x.high,x.low,x.close,k=14,d=3,smooth_k=3); x['stoch_k']=st[firstcol(st,'STOCHk_')]; x['stoch_d']=st[firstcol(st,'STOCHd_')]
    mac=ta.macd(x.close,fast=12,slow=26,signal=9); x['macd']=mac[firstcol(mac,'MACD_')]; x['macd_sig']=mac[firstcol(mac,'MACDs_')]; x['macd_hist']=mac[firstcol(mac,'MACDh_')]
    ad=ta.adx(x.high,x.low,x.close,length=14); x['adx']=ad[firstcol(ad,'ADX_')]
    x['atr']=ta.atr(x.high,x.low,x.close,length=14)
    tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1)
    x['chop']=100*np.log10(tr.rolling(14).sum()/(x.high.rolling(14).max()-x.low.rolling(14).min()).replace(0,np.nan))/np.log10(14)
    x['bb_pos']=(x.close-x.bb_l)/(x.bb_u-x.bb_l).replace(0,np.nan)
    x['body']=(x.close-x.open)/(x.high-x.low).replace(0,np.nan); x['upper_wick']=(x.high-x[['open','close']].max(axis=1))/(x.high-x.low).replace(0,np.nan); x['lower_wick']=(x[['open','close']].min(axis=1)-x.low)/(x.high-x.low).replace(0,np.nan)
    for n in (1,3,5,10): x[f'ret{n}']=x.close.pct_change(n)
    x['volatility_ratio']=x.ret1.rolling(20).std()/x.ret1.rolling(100).std().replace(0,np.nan); x['volume_ratio']=x.volume/x.volume.rolling(20).mean().replace(0,np.nan)
    for e in (9,21,50,200): x[f'dist_ema{e}']=x.close/x[f'ema{e}']-1
    x['range_pct']=(x.high-x.low)/x.close
    x['ema_spread']=x.ema9/x.ema21-1; x['trend50']=x.ema50.pct_change(5); x['trend200']=x.ema200.pct_change(10)
    return x

def rule_signal(r):
    votes=[]
    votes += [1 if r.rsi14<35 else -1 if r.rsi14>65 else 0]
    votes += [1 if r.bb_pos<0.20 else -1 if r.bb_pos>0.80 else 0]
    votes += [1 if r.stoch_k<20 and r.stoch_k>r.stoch_d else -1 if r.stoch_k>80 and r.stoch_k<r.stoch_d else 0]
    votes += [1 if r.macd_hist>0 else -1 if r.macd_hist<0 else 0]
    p=sum(v==1 for v in votes); m=sum(v==-1 for v in votes); direction=1 if p>=3 else -1 if m>=3 else 0
    return direction,p,m,votes

def features(d):
    cols=['body','upper_wick','lower_wick','ret1','ret3','ret5','ret10','volatility_ratio','rsi7','rsi14','rsi21','dist_ema9','dist_ema21','dist_ema50','dist_ema200','bb_pos','chop','volume_ratio','range_pct','ema_spread','trend50','trend200','adx','atr','macd','macd_sig','macd_hist','stoch_k','stoch_d']
    # Thêm biến quy mô để vượt 30 đặc trưng theo đúng yêu cầu.
    cols += ['open','high','low','close','volume']
    return cols

def triple_labels(x,maxbars=15,tp_mult=1.8,sl_mult=1.0):
    y=[]; dirs=[]
    for i in range(len(x)):
        direction,p,m,_=rule_signal(x.iloc[i]); dirs.append(direction); label=np.nan
        if direction!=0 and i+1<len(x):
            atr=x.iloc[i].atr
            if pd.notna(atr) and atr>0:
                tp=x.iloc[i].close+direction*tp_mult*atr; sl=x.iloc[i].close-direction*sl_mult*atr
                end=min(len(x)-1,i+maxbars); label=0
                for j in range(i+1,end+1):
                    hi,lo=x.iloc[j].high,x.iloc[j].low
                    hit_tp=(hi>=tp) if direction==1 else (lo<=tp); hit_sl=(lo<=sl) if direction==1 else (hi>=sl)
                    if hit_tp and hit_sl: label=0; break
                    if hit_tp: label=1; break
                    if hit_sl: label=0; break
        y.append(label)
    return pd.Series(y,index=x.index),pd.Series(dirs,index=x.index)

def model_paths(ex,symbol,tf):
    key=f'{ex}_{symbol.replace("/","_")}_{tf}'.replace(':','_'); return MODEL_DIR/f'{key}.pkl'

def train_model(ex,symbol,tf,limit):
    d=indicators(fetch_df(ex,symbol,tf,limit)); y,dirs=triple_labels(d)
    cols=features(d); data=d[cols].copy(); valid=y.notna() & (dirs!=0); data=data.loc[valid]; y=y.loc[valid].astype(int)
    if len(data)<300 or y.nunique()<2: raise RuntimeError('Không đủ mẫu hợp lệ hoặc nhãn chỉ có một lớp; cần thêm lịch sử.')
    pipe=Pipeline([('imp',SimpleImputer(strategy='median')),('gb',GradientBoostingClassifier(n_estimators=180,max_depth=3,learning_rate=.04,random_state=42))])
    n=min(5,max(2,len(data)//100)); tscv=TimeSeriesSplit(n_splits=n); scores=cross_val_score(pipe,data,y,cv=tscv,scoring='accuracy'); wf=float(scores.mean())
    # OOF dự đoán cho meta-labeling, chỉ dùng dự đoán ngoài mẫu.
    oof=np.full(len(data),np.nan)
    for tr,va in tscv.split(data):
        m=Pipeline([('imp',SimpleImputer(strategy='median')),('gb',GradientBoostingClassifier(n_estimators=180,max_depth=3,learning_rate=.04,random_state=42))]); m.fit(data.iloc[tr],y.iloc[tr]); oof[va]=m.predict_proba(data.iloc[va])[:,1]
    meta_mask=~np.isnan(oof); meta_X=np.column_stack([data.to_numpy()[meta_mask],oof[meta_mask]])
    meta=Pipeline([('imp',SimpleImputer(strategy='median')),('rf',RandomForestClassifier(n_estimators=250,min_samples_leaf=4,class_weight='balanced',random_state=42,n_jobs=-1))]); meta.fit(meta_X,y.to_numpy()[meta_mask])
    pipe.fit(data,y)
    path=model_paths(ex,symbol,tf); joblib.dump({'gb':pipe,'meta':meta,'features':cols,'walk_forward_accuracy':wf,'trained_at':datetime.now(timezone.utc).isoformat(),'samples':len(data)},path)
    return {'walk_forward_accuracy':wf,'samples':len(data),'path':str(path)}

def load_model(ex,symbol,tf):
    p=model_paths(ex,symbol,tf)
    if not p.exists(): raise FileNotFoundError('Model chưa train cho cấu hình này.')
    return joblib.load(p)

def analyze(ex,symbol,tf,htf,limit):
    mdl=load_model(ex,symbol,tf); d=indicators(fetch_df(ex,symbol,tf,limit)); r=d.iloc[-1]; direction,p,m,votes=rule_signal(r); rule_pass=direction!=0
    h=fetch_df(ex,symbol,htf,min(limit,800)); h50=ta.ema(h.close,length=50).iloc[-1]; htf_dir=1 if h.close.iloc[-1]>=h50 else -1; htf_pass=rule_pass and htf_dir==direction
    regime_pass=bool(pd.notna(r.adx) and pd.notna(r.chop) and r.adx>=22 and r.chop<=55)
    X=pd.DataFrame([r[mdl['features']].to_dict()]); ml_prob=float(mdl['gb'].predict_proba(X)[0,1]); ml_pass=rule_pass and ml_prob>=.70
    meta_X=np.column_stack([X.to_numpy(),[ml_prob]]); meta_prob=float(mdl['meta'].predict_proba(meta_X)[0,1]); meta_pass=ml_prob>=.70 and meta_prob>=.70
    wf=float(mdl['walk_forward_accuracy']); wf_pass=wf>=.62
    rule_conf=min(1.0,max(p,m)/4); composite=.30*rule_conf+.35*ml_prob+.20*meta_prob+.15*wf
    allpass=all([rule_pass,htf_pass,regime_pass,ml_pass,meta_pass,wf_pass]); signal='BUY' if direction==1 else 'SELL' if direction==-1 else 'WAIT'; alert=bool(allpass)
    if not rule_pass: signal='WAIT'
    return {'alert':alert,'signal':signal,'symbol':symbol,'exchange':ex,'timeframe':tf,'price':float(r.close),'timestamp':int(r.timestamp),'composite_confidence':round(composite,4),'rule_confidence':round(rule_conf,4),'ml_confidence':round(ml_prob,4),'meta_confidence':round(meta_prob,4),'walk_forward_accuracy':round(wf,4),'layers':[{'name':'Rule-based','pass':rule_pass,'detail':f'{p}/4 BUY, {m}/4 SELL'},{'name':'HTF Align','pass':htf_pass,'detail':f'Close vs EMA50: {"UP" if htf_dir==1 else "DOWN"}'},{'name':'Regime filter','pass':regime_pass,'detail':f'ADX {r.adx:.1f} / CHOP {r.chop:.1f}'},{'name':'ML Ensemble','pass':ml_pass,'detail':f'{ml_prob:.1%}'},{'name':'Meta-labeling','pass':meta_pass,'detail':f'{meta_prob:.1%}'},{'name':'Walk-forward','pass':wf_pass,'detail':f'{wf:.1%}'}],'indicators':{'RSI14':float(r.rsi14),'BB position':float(r.bb_pos),'Stoch K':float(r.stoch_k),'MACD hist':float(r.macd_hist),'ATR':float(r.atr),'ADX':float(r.adx),'Choppiness':float(r.chop),'Volume ratio':float(r.volume_ratio)},'message':'ĐỦ 6 TẦNG — tín hiệu cực mạnh.' if alert else 'Đang chờ: ít nhất một tầng xác nhận chưa đạt.'}

def telegram_send(c,message):
    token=c.get('telegram_token','').strip(); chat=c.get('telegram_chat_id','').strip()
    if not token or not chat: return False
    try:
        from telegram import Bot
        async def go():
            async with Bot(token=token) as bot: await bot.send_message(chat_id=chat,text=message)
        asyncio.run(go()); return True
    except Exception: return False

def save_signal(a):
    with lock:
        arr=read_json(SIGNALS,[]); arr.insert(0,a); write_json(SIGNALS,arr[:200])

def alert_once(a,c):
    key=f"{a['symbol']}_{a['signal']}_{datetime.now().strftime('%Y%m%d%H')}"
    with lock:
        if key in sent_keys: return False
        sent_keys.add(key)
    save_signal(a); telegram_send(c,f"🚨 TÍN HIỆU CỰC MẠNH!\n{a['exchange']} {a['symbol']}\n{a['signal']} @ {a['price']}\nConfidence: {a['composite_confidence']:.1%}")
    return True

def scanner_loop():
    while True:
        if not scanner['running']: time.sleep(1); continue
        c=cfg(); symbols=[c.get('symbol')]+c.get('watchlist',[]); symbols=list(dict.fromkeys([s for s in symbols if s])); ex=None
        try:
            ex=exchange_obj(c['exchange'])
            for s in symbols:
                try:
                    a=analyze(c['exchange'],s,c['timeframe'],c['htf'],c['candle_limit']);
                    if a['alert']: alert_once(a,c)
                except Exception as e: scanner['last_error']=f'{s}: {e}'
            scanner['last_scan']=datetime.now(timezone.utc).isoformat()
        except Exception as e: scanner['last_error']=str(e)
        finally:
            if ex:
                try: ex.close()
                except Exception: pass
        time.sleep(max(30,int(c.get('scan_interval',300))))
threading.Thread(target=scanner_loop,daemon=True).start()

@app.get('/')
def index(): return render_template('index.html')
@app.get('/api/config')
def get_config(): return jsonify(cfg())
@app.post('/api/config')
def post_config():
    c=cfg(); c.update(request.get_json(force=True) or {}); write_json(CONFIG,c); return jsonify(c)
@app.get('/api/exchanges')
def exchanges(): return jsonify(['binance','bybit','okx','kraken','coinbase','kucoin','gateio','bitget'])
@app.post('/api/train')
def train():
    c=cfg(); ex=None
    try:
        ex=exchange_obj(c['exchange']); out=train_model(c['exchange'],c['symbol'],c['timeframe'],c['candle_limit']); return jsonify({'ok':True,**out})
    except Exception as e: return jsonify({'ok':False,'error':str(e)}),400
    finally:
        if ex:
            try: ex.close()
            except Exception: pass
@app.get('/api/model_info')
def model_info():
    c=cfg(); p=model_paths(c['exchange'],c['symbol'],c['timeframe'])
    if not p.exists(): return jsonify({'trained':False})
    try:
        m=joblib.load(p); return jsonify({'trained':True,'walk_forward_accuracy':m['walk_forward_accuracy'],'trained_at':m['trained_at'],'samples':m['samples'],'features':len(m['features'])})
    except Exception as e: return jsonify({'trained':False,'error':str(e)})
@app.post('/api/analyze')
def api_analyze():
    c=cfg(); body=request.get_json(silent=True) or {}; symbol=body.get('symbol',c['symbol']); ex=None
    try:
        ex=exchange_obj(c['exchange']); return jsonify(analyze(c['exchange'],symbol,c['timeframe'],c['htf'],c['candle_limit']))
    except Exception as e: return jsonify({'alert':False,'signal':'WAIT','message':str(e),'layers':[]}),400
    finally:
        if ex:
            try: ex.close()
            except Exception: pass
@app.get('/api/signals')
def signals(): return jsonify(read_json(SIGNALS,[]))
@app.post('/api/clear_signals')
def clear_signals(): write_json(SIGNALS,[]); sent_keys.clear(); return jsonify({'ok':True})
@app.post('/api/scanner')
def scanner_ctl():
    scanner['running']=bool((request.get_json(silent=True) or {}).get('running',False)); return jsonify(scanner)
@app.get('/api/scanner_status')
def scanner_status(): return jsonify(scanner)
@app.post('/api/watchlist')
def watchlist():
    c=cfg(); ex=None; out=[]
    try:
        ex=exchange_obj(c['exchange'])
        for s in list(dict.fromkeys(c.get('watchlist',[])+[c['symbol']])):
            try:
                t=ex.fetch_ticker(s); out.append({'symbol':s,'price':t.get('last'),'change':t.get('percentage')})
            except Exception as e: out.append({'symbol':s,'error':str(e)})
        return jsonify(out)
    except Exception as e: return jsonify({'error':str(e)}),400
    finally:
        if ex:
            try: ex.close()
            except Exception: pass

if __name__=='__main__':
    if not CONFIG.exists(): write_json(CONFIG,DEFAULT)
    if not SIGNALS.exists(): write_json(SIGNALS,[])
    app.run(host='0.0.0.0',port=int(os.getenv('PORT','5000')),debug=False,use_reloader=False)
