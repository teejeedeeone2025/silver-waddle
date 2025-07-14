import ccxt
import pandas as pd
import pytz
import numpy as np
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Initialize the exchange
exchange = ccxt.bitget()

# List of symbols to check
symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'XRP/USDT:USDT', 'BCH/USDT:USDT', 'LTC/USDT:USDT', 'ADA/USDT:USDT', 'ETC/USDT:USDT', 'LINK/USDT:USDT', 'TRX/USDT:USDT', 'DOT/USDT:USDT', 'DOGE/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT', 'UNI/USDT:USDT', 'ICP/USDT:USDT', 'AAVE/USDT:USDT', 'FIL/USDT:USDT', 'XLM/USDT:USDT', 'ATOM/USDT:USDT', 'XTZ/USDT:USDT', 'SUSHI/USDT:USDT', 'AXS/USDT:USDT', 'THETA/USDT:USDT', 'AVAX/USDT:USDT', 'SHIB/USDT:USDT', 'MANA/USDT:USDT', 'GALA/USDT:USDT', 'SAND/USDT:USDT', 'DYDX/USDT:USDT', 'CRV/USDT:USDT', 'NEAR/USDT:USDT', 'EGLD/USDT:USDT', 'KSM/USDT:USDT', 'AR/USDT:USDT', 'PEOPLE/USDT:USDT', 'LRC/USDT:USDT', 'NEO/USDT:USDT', 'ALICE/USDT:USDT', 'WAVES/USDT:USDT', 'ALGO/USDT:USDT', 'IOTA/USDT:USDT', 'ENJ/USDT:USDT', 'GMT/USDT:USDT', 'ZIL/USDT:USDT', 'IOST/USDT:USDT', 'APE/USDT:USDT', 'RUNE/USDT:USDT', 'KNC/USDT:USDT', 'APT/USDT:USDT', 'CHZ/USDT:USDT', 'ROSE/USDT:USDT', 'ZRX/USDT:USDT', 'KAVA/USDT:USDT', 'ENS/USDT:USDT', 'MTL/USDT:USDT', 'AUDIO/USDT:USDT', 'SXP/USDT:USDT', 'C98/USDT:USDT', 'OP/USDT:USDT', 'RSR/USDT:USDT', 'SNX/USDT:USDT', 'STORJ/USDT:USDT', '1INCH/USDT:USDT', 'COMP/USDT:USDT', 'IMX/USDT:USDT', 'LUNA/USDT:USDT', 'FLOW/USDT:USDT', 'TRB/USDT:USDT', 'QTUM/USDT:USDT', 'API3/USDT:USDT', 'MASK/USDT:USDT', 'WOO/USDT:USDT', 'GRT/USDT:USDT', 'BAND/USDT:USDT', 'STG/USDT:USDT', 'LUNC/USDT:USDT', 'ONE/USDT:USDT', 'JASMY/USDT:USDT', 'MKR/USDT:USDT', 'BAT/USDT:USDT', 'MAGIC/USDT:USDT', 'ALPHA/USDT:USDT', 'LDO/USDT:USDT', 'CELO/USDT:USDT', 'BLUR/USDT:USDT', 'MINA/USDT:USDT', 'CORE/USDT:USDT', 'CFX/USDT:USDT', 'ASTR/USDT:USDT', 'GMX/USDT:USDT', 'ANKR/USDT:USDT', 'ACH/USDT:USDT', 'FET/USDT:USDT', 'FXS/USDT:USDT', 'HOOK/USDT:USDT', 'SSV/USDT:USDT', 'USDC/USDT:USDT', 'LQTY/USDT:USDT', 'STX/USDT:USDT', 'TRU/USDT:USDT', 'HBAR/USDT:USDT', 'INJ/USDT:USDT', 'BEL/USDT:USDT', 'COTI/USDT:USDT', 'VET/USDT:USDT', 'ARB/USDT:USDT', 'LOOKS/USDT:USDT', 'KAIA/USDT:USDT', 'FLM/USDT:USDT', 'CKB/USDT:USDT', 'ID/USDT:USDT', 'JOE/USDT:USDT', 'TLM/USDT:USDT', 'HOT/USDT:USDT', 'CHR/USDT:USDT', 'RDNT/USDT:USDT', 'ICX/USDT:USDT', 'HFT/USDT:USDT', 'ONT/USDT:USDT', 'NKN/USDT:USDT', 'ARPA/USDT:USDT', 'SFP/USDT:USDT', 'CTSI/USDT:USDT', 'SKL/USDT:USDT', 'RVN/USDT:USDT', 'CELR/USDT:USDT', 'FLOKI/USDT:USDT', 'SPELL/USDT:USDT', 'SUI/USDT:USDT', 'PEPE/USDT:USDT', 'IOTX/USDT:USDT', 'CTK/USDT:USDT', 'UMA/USDT:USDT', 'TURBO/USDT:USDT', 'BSV/USDT:USDT', 'TON/USDT:USDT', 'GTC/USDT:USDT', 'DENT/USDT:USDT', 'ZEN/USDT:USDT', 'PHB/USDT:USDT', 'ORDI/USDT:USDT', '1000BONK/USDT:USDT', 'LEVER/USDT:USDT', 'USTC/USDT:USDT', 'RAD/USDT:USDT', 'QNT/USDT:USDT', 'MAV/USDT:USDT', 'XVG/USDT:USDT', '1000XEC/USDT:USDT', 'AGLD/USDT:USDT', 'WLD/USDT:USDT', 'PENDLE/USDT:USDT', 'ARKM/USDT:USDT', 'CVX/USDT:USDT', 'YGG/USDT:USDT', 'OGN/USDT:USDT', 'LPT/USDT:USDT', 'BNT/USDT:USDT', 'SEI/USDT:USDT', 'CYBER/USDT:USDT', 'BAKE/USDT:USDT', 'BIGTIME/USDT:USDT', 'WAXP/USDT:USDT', 'POLYX/USDT:USDT', 'TIA/USDT:USDT', 'MEME/USDT:USDT', 'PYTH/USDT:USDT', 'JTO/USDT:USDT', '1000SATS/USDT:USDT', '1000RATS/USDT:USDT', 'ACE/USDT:USDT', 'XAI/USDT:USDT', 'MANTA/USDT:USDT', 'ALT/USDT:USDT', 'JUP/USDT:USDT', 'ZETA/USDT:USDT', 'STRK/USDT:USDT', 'PIXEL/USDT:USDT', 'DYM/USDT:USDT', 'WIF/USDT:USDT', 'AXL/USDT:USDT', 'BEAM/USDT:USDT', 'BOME/USDT:USDT', 'METIS/USDT:USDT', 'NFP/USDT:USDT', 'VANRY/USDT:USDT', 'AEVO/USDT:USDT', 'ETHFI/USDT:USDT', 'OM/USDT:USDT', 'ONDO/USDT:USDT', 'CAKE/USDT:USDT', 'PORTAL/USDT:USDT', 'NTRN/USDT:USDT', 'KAS/USDT:USDT', 'AI/USDT:USDT', 'ENA/USDT:USDT', 'W/USDT:USDT', 'CVC/USDT:USDT', 'TNSR/USDT:USDT', 'SAGA/USDT:USDT', 'TAO/USDT:USDT', 'RAY/USDT:USDT', 'ATA/USDT:USDT', 'SUPER/USDT:USDT', 'ONG/USDT:USDT', 'OMNI1/USDT:USDT', 'LSK/USDT:USDT', 'GLM/USDT:USDT', 'REZ/USDT:USDT', 'XVS/USDT:USDT', 'MOVR/USDT:USDT', 'BB/USDT:USDT', 'NOT/USDT:USDT', 'BICO/USDT:USDT', 'HIFI/USDT:USDT', 'IO/USDT:USDT', 'TAIKO/USDT:USDT', 'BRETT/USDT:USDT', 'ATH/USDT:USDT', 'ZK/USDT:USDT', 'MEW/USDT:USDT', 'LISTA/USDT:USDT', 'ZRO/USDT:USDT', 'BLAST/USDT:USDT', 'DOG/USDT:USDT', 'PAXG/USDT:USDT', 'ZKJ/USDT:USDT', 'BGB/USDT:USDT', 'MOCA/USDT:USDT', 'GAS/USDT:USDT', 'UXLINK/USDT:USDT', 'BANANA/USDT:USDT', 'MYRO/USDT:USDT', 'POPCAT/USDT:USDT', 'PRCL/USDT:USDT', 'AVAIL/USDT:USDT', 'RENDER/USDT:USDT', 'RARE/USDT:USDT', 'PONKE/USDT:USDT', 'T/USDT:USDT', '1000000MOG/USDT:USDT', 'G/USDT:USDT', 'SYN/USDT:USDT', 'SYS/USDT:USDT', 'VOXEL/USDT:USDT', 'SUN/USDT:USDT', 'DOGS/USDT:USDT', 'ORDER/USDT:USDT', 'SUNDOG/USDT:USDT', 'AKT/USDT:USDT', 'MBOX/USDT:USDT', 'HNT/USDT:USDT', 'CHESS/USDT:USDT', 'FLUX/USDT:USDT', 'POL/USDT:USDT', 'BSW/USDT:USDT', 'NEIROETH/USDT:USDT', 'RPL/USDT:USDT', 'QUICK/USDT:USDT', 'AERGO/USDT:USDT', '1MBABYDOGE/USDT:USDT', '1000CAT/USDT:USDT', 'KDA/USDT:USDT', 'FIDA/USDT:USDT', 'CATI/USDT:USDT', 'FIO/USDT:USDT', 'ARK/USDT:USDT', 'GHST/USDT:USDT', 'LOKA/USDT:USDT', 'VELO/USDT:USDT', 'HMSTR/USDT:USDT', 'AGI/USDT:USDT', 'REI/USDT:USDT', 'COS/USDT:USDT', 'EIGEN/USDT:USDT', 'MOODENG/USDT:USDT', 'DIA/USDT:USDT', 'OG/USDT:USDT', 'NEIROCTO/USDT:USDT', 'ETHW/USDT:USDT', 'DegenReborn/USDT:USDT', 'KMNO/USDT:USDT', 'POWR/USDT:USDT', 'PYR/USDT:USDT', 'CARV/USDT:USDT', 'SLERF/USDT:USDT', 'PUFFER/USDT:USDT', '10000WHY/USDT:USDT', 'DEEP/USDT:USDT', 'DBR/USDT:USDT', 'LUMIA/USDT:USDT', 'SCR/USDT:USDT', 'GOAT/USDT:USDT', 'X/USDT:USDT', 'SAFE/USDT:USDT', 'GRASS/USDT:USDT', 'SWEAT/USDT:USDT', 'SANTOS/USDT:USDT', 'SPX/USDT:USDT', 'VIRTUAL/USDT:USDT', 'AERO/USDT:USDT', 'CETUS/USDT:USDT', 'COW/USDT:USDT', 'SWELL/USDT:USDT', 'DRIFT/USDT:USDT', 'PNUT/USDT:USDT', 'ACT/USDT:USDT', 'CRO/USDT:USDT', 'PEAQ/USDT:USDT', 'FWOG/USDT:USDT', 'HIPPO/USDT:USDT', 'SNT/USDT:USDT', 'MERL/USDT:USDT', 'STEEM/USDT:USDT', 'BAN/USDT:USDT', 'OL/USDT:USDT', 'MORPHO/USDT:USDT', 'SCRT/USDT:USDT', 'CHILLGUY/USDT:USDT', '1MCHEEMS/USDT:USDT', 'OXT/USDT:USDT', 'ZRC/USDT:USDT', 'THE/USDT:USDT', 'MAJOR/USDT:USDT', 'CTC/USDT:USDT', 'XDC/USDT:USDT', 'XION/USDT:USDT', 'ORCA/USDT:USDT', 'ACX/USDT:USDT', 'NS/USDT:USDT', 'MOVE/USDT:USDT', 'KOMA/USDT:USDT', 'ME/USDT:USDT', 'VELODROME/USDT:USDT', 'AVA/USDT:USDT', 'VANA/USDT:USDT', 'HYPE/USDT:USDT', 'PENGU/USDT:USDT', 'USUAL/USDT:USDT', 'FUEL/USDT:USDT', 'CGPT/USDT:USDT', 'AIXBT/USDT:USDT', 'FARTCOIN/USDT:USDT', 'HIVE/USDT:USDT', 'DEXE/USDT:USDT', 'GIGA/USDT:USDT', 'PHA/USDT:USDT', 'DF/USDT:USDT', 'AI16Z/USDT:USDT', 'GRIFFAIN/USDT:USDT', 'ZEREBRO/USDT:USDT', 'BIO/USDT:USDT', 'SWARMS/USDT:USDT', 'ALCH/USDT:USDT', 'COOKIE/USDT:USDT', 'SONIC/USDT:USDT', 'AVAAI/USDT:USDT', 'S/USDT:USDT', 'PROM/USDT:USDT', 'DUCK/USDT:USDT', 'BGSC/USDT:USDT', 'SOLV/USDT:USDT', 'ARC/USDT:USDT', 'PIPPIN/USDT:USDT', 'TRUMP/USDT:USDT', 'MELANIA/USDT:USDT', 'PLUME/USDT:USDT', 'VTHO/USDT:USDT', 'J/USDT:USDT', 'VINE/USDT:USDT', 'ANIME/USDT:USDT', 'XCN/USDT:USDT', 'TOSHI/USDT:USDT', 'VVV/USDT:USDT', 'FORTH/USDT:USDT', 'BERA/USDT:USDT', 'TSTBSC/USDT:USDT', '10000ELON/USDT:USDT', 'LAYER/USDT:USDT', 'B3/USDT:USDT', 'IP/USDT:USDT', 'RON/USDT:USDT', 'HEI/USDT:USDT', 'SHELL/USDT:USDT', 'BROCCOLI/USDT:USDT', 'AUCTION/USDT:USDT', 'GPS/USDT:USDT', 'GNO/USDT:USDT', 'AIOZ/USDT:USDT', 'PI/USDT:USDT', 'AVL/USDT:USDT', 'KAITO/USDT:USDT', 'GODS/USDT:USDT', 'ROAM/USDT:USDT', 'RED/USDT:USDT', 'ELX/USDT:USDT', 'SERAPH/USDT:USDT', 'BMT/USDT:USDT', 'VIC/USDT:USDT', 'EPIC/USDT:USDT', 'OBT/USDT:USDT', 'MUBARAK/USDT:USDT', 'NMR/USDT:USDT', 'TUT/USDT:USDT', 'FORM/USDT:USDT', 'RSS3/USDT:USDT', 'BID/USDT:USDT', 'SIREN/USDT:USDT', 'BANANAS31/USDT:USDT', 'BR/USDT:USDT', 'NIL/USDT:USDT', 'PARTI/USDT:USDT', 'NAVX/USDT:USDT', 'WAL/USDT:USDT', 'KILO/USDT:USDT', 'FUN/USDT:USDT', 'MLN/USDT:USDT', 'GUN/USDT:USDT', 'PUMPBTC/USDT:USDT', 'STO/USDT:USDT', 'XAUT/USDT:USDT', 'AMP/USDT:USDT', 'BABY/USDT:USDT', 'FHE/USDT:USDT', 'PROMPT/USDT:USDT', 'RFC/USDT:USDT', 'KERNEL/USDT:USDT', 'WCT/USDT:USDT', '10000000AIDOGE/USDT:USDT', 'BANK/USDT:USDT', 'EPT/USDT:USDT', 'HYPER/USDT:USDT', 'ZORA/USDT:USDT', 'INIT/USDT:USDT', 'DOLO/USDT:USDT', 'FIS/USDT:USDT', 'JST/USDT:USDT', 'TAI/USDT:USDT', 'SIGN/USDT:USDT', 'MILK/USDT:USDT', 'HAEDAL/USDT:USDT', 'PUNDIX/USDT:USDT', 'B2/USDT:USDT', 'GORK/USDT:USDT', 'HOUSE/USDT:USDT', 'ASR/USDT:USDT', 'ALPINE/USDT:USDT', 'SYRUP/USDT:USDT', 'OBOL/USDT:USDT', 'SXT/USDT:USDT', 'SHM/USDT:USDT', 'DOOD/USDT:USDT', 'SKYAI/USDT:USDT', 'LAUNCHCOIN/USDT:USDT', 'NXPC/USDT:USDT', 'BADGER/USDT:USDT', 'AGT/USDT:USDT', 'AWE/USDT:USDT', 'TGT/USDT:USDT', 'RWA/USDT:USDT', 'BLUE/USDT:USDT', 'B/USDT:USDT', 'SOON/USDT:USDT', 'ZBCN/USDT:USDT', 'HUMA/USDT:USDT', 'PFVS/USDT:USDT', 'SOPH/USDT:USDT', 'ELDE/USDT:USDT', 'A/USDT:USDT', 'ASRR/USDT:USDT', 'BDXN/USDT:USDT', 'PORT3/USDT:USDT', 'LA/USDT:USDT', 'CUDIS/USDT:USDT', 'SKATE/USDT:USDT', 'RESOLV/USDT:USDT', 'HOME/USDT:USDT', 'IDOL/USDT:USDT', 'SQD/USDT:USDT', 'SPK/USDT:USDT', 'BOMB/USDT:USDT', 'F/USDT:USDT', 'NEWT/USDT:USDT', 'DMC/USDT:USDT', 'H/USDT:USDT', 'SAHARA/USDT:USDT', 'NODE/USDT:USDT', 'FRAG/USDT:USDT', 'ICNT/USDT:USDT', 'M/USDT:USDT', 'CBK/USDT:USDT', 'CROSS/USDT:USDT', 'BULLA/USDT:USDT', 'TANSSI/USDT:USDT', 'AIN/USDT:USDT']


# Parameters
timeframe_15m = '15m'
timeframe_1h = '1h'
limit = 500
ema_fast_length = 38
ema_slow_length = 62
ema_trend_length = 200

# NWE Parameters
h = 8.0          # Bandwidth
mult = 3.0       # Multiplier for MAE
repaint = True   # Repainting mode

# Email settings
SENDER_EMAIL = "dahmadu071@gmail.com"
RECIPIENT_EMAILS = ["teejeedeeone@gmail.com"]
EMAIL_PASSWORD = "oase wivf hvqn lyhr"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Set the timezone
lagos_tz = pytz.timezone('Africa/Lagos')

def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = ", ".join(RECIPIENT_EMAILS)
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, EMAIL_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
        server.quit()
        print("Email notification sent successfully")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

# Gaussian window function
def gauss(x, h):
    return np.exp(-(x ** 2) / (h ** 2 * 2))

# Calculate Nadaraya-Watson Envelope
def calculate_nwe(src, h, mult, repaint):
    n = len(src)
    out = np.zeros(n)
    mae = np.zeros(n)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    if not repaint:
        # Non-repainting mode
        coefs = np.array([gauss(i, h) for i in range(n)])
        den = np.sum(coefs)
        
        for i in range(n):
            out[i] = np.sum(src * coefs) / den
        
        mae = pd.Series(np.abs(src - out)).rolling(499).mean().values * mult
        upper = out + mae
        lower = out - mae
    else:
        # Repainting mode
        nwe = []
        sae = 0.0
        
        for i in range(n):
            sum_val = 0.0
            sumw = 0.0
            for j in range(n):
                w = gauss(i - j, h)
                sum_val += src[j] * w
                sumw += w
            y2 = sum_val / sumw
            nwe.append(y2)
            sae += np.abs(src[i] - y2)
        
        sae = (sae / n) * mult
        
        for i in range(n):
            upper[i] = nwe[i] + sae
            lower[i] = nwe[i] - sae
            out[i] = nwe[i]
    
    return out, upper, lower

# Function to fetch historical data
def fetch_data(symbol, timeframe, limit):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(lagos_tz)
    df.set_index('timestamp', inplace=True)
    return df

# Function to check for valid signals on the last closed candle
def check_valid_signal(symbol):
    try:
        df_15m = fetch_data(symbol, timeframe_15m, limit)
        
        # Calculate NWE for crossunder/crossover detection
        src = df_15m['close'].values
        _, upper, lower = calculate_nwe(src, h, mult, repaint)
        df_15m['upper'] = upper
        df_15m['lower'] = lower
        
        # Detect crossings
        df_15m['crossunder'] = (df_15m['close'].shift(1) > df_15m['lower'].shift(1)) & (df_15m['close'] < df_15m['lower'])
        df_15m['crossover'] = (df_15m['close'].shift(1) < df_15m['upper'].shift(1)) & (df_15m['close'] > df_15m['upper'])
        
        df_1h = fetch_data(symbol, timeframe_1h, limit)

        # Calculate EMAs
        df_15m['EMA_Fast'] = df_15m['close'].ewm(span=ema_fast_length, adjust=False).mean()
        df_15m['EMA_Slow'] = df_15m['close'].ewm(span=ema_slow_length, adjust=False).mean()
        df_15m['EMA_Trend'] = df_15m['close'].ewm(span=ema_trend_length, adjust=False).mean()

        # Calculate 1h EMA Trend
        df_1h['EMA_Trend'] = df_1h['close'].ewm(span=ema_trend_length, adjust=False).mean()
        df_1h_resampled = df_1h['EMA_Trend'].resample('15min').ffill()
        df_15m['EMA_Trend_1h'] = df_1h_resampled

        # Generate primary signals
        df_15m['Primary_Buy'] = (df_15m['EMA_Fast'] > df_15m['EMA_Slow']) & \
                               (df_15m['EMA_Fast'].shift(1) <= df_15m['EMA_Slow'].shift(1)) & \
                               (df_15m['close'] > df_15m['EMA_Trend']) & \
                               (df_15m['close'] > df_15m['EMA_Trend_1h'])

        df_15m['Primary_Sell'] = (df_15m['EMA_Fast'] < df_15m['EMA_Slow']) & \
                                (df_15m['EMA_Fast'].shift(1) >= df_15m['EMA_Slow'].shift(1)) & \
                                (df_15m['close'] < df_15m['EMA_Trend']) & \
                                (df_15m['close'] < df_15m['EMA_Trend_1h'])

        # Conservative entry conditions
        df_15m['Conservative_Up'] = (df_15m['EMA_Fast'] > df_15m['EMA_Slow']) & \
                                   (df_15m['close'].shift(1) < df_15m['EMA_Fast']) & \
                                   (df_15m['close'] > df_15m['EMA_Fast'])

        df_15m['Conservative_Dn'] = (df_15m['EMA_Fast'] < df_15m['EMA_Slow']) & \
                                   (df_15m['close'].shift(1) > df_15m['EMA_Fast']) & \
                                   (df_15m['close'] < df_15m['EMA_Fast'])

        # Track signals with state variables
        class SignalTracker:
            def __init__(self):
                self.last_primary = None
                self.conservative_plotted = False
                self.valid_primary_signals = []
                self.valid_conservative_signals = []

            def process_row(self, index, row):
                # Check for new primary signals
                if row['Primary_Buy']:
                    self.last_primary = ('buy', index, row['close'])
                    self.conservative_plotted = False
                elif row['Primary_Sell']:
                    self.last_primary = ('sell', index, row['close'])
                    self.conservative_plotted = False

                # Check for conservative signals (only the first one after primary)
                if not self.conservative_plotted and self.last_primary:
                    if self.last_primary[0] == 'buy' and row['Conservative_Up']:
                        self.valid_conservative_signals.append((index, 'Conservative Buy', row['close']))
                        self.valid_primary_signals.append((self.last_primary[1], f"Primary Buy", self.last_primary[2]))
                        self.conservative_plotted = True
                    elif self.last_primary[0] == 'sell' and row['Conservative_Dn']:
                        self.valid_conservative_signals.append((index, 'Conservative Sell', row['close']))
                        self.valid_primary_signals.append((self.last_primary[1], f"Primary Sell", self.last_primary[2]))
                        self.conservative_plotted = True

        # Process all rows to track signals
        tracker = SignalTracker()
        for index, row in df_15m.iterrows():
            tracker.process_row(index, row)

        # Get the last two closed candles
        last_candle = df_15m.iloc[-2]
        prev_last_candle = df_15m.iloc[-3]

        # Check if the last closed candle has a valid conservative signal
        valid_signal = None
        for signal in tracker.valid_conservative_signals:
            if signal[0] == last_candle.name:  # Check timestamp
                # Additional validation for crossunder/crossover
                if "Buy" in signal[1]:
                    # For buy signals, require crossunder in last 2 candles
                    if (last_candle['crossunder'] or prev_last_candle['crossunder']):
                        valid_signal = signal[1]
                elif "Sell" in signal[1]:
                    # For sell signals, require crossover in last 2 candles
                    if (last_candle['crossover'] or prev_last_candle['crossover']):
                        valid_signal = signal[1]
                break

        # Print the result and send email if valid signal
        current_time = datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')
        if valid_signal:
            message = f"{symbol} - {valid_signal} at {current_time} (Lagos Time)"
            print(message)
            
            # Send email notification
            email_subject = f"Crypto Alert: {symbol} {valid_signal}"
            email_body = f"""
            Trading Signal Detected:
            
            Symbol: {symbol}
            Signal: {valid_signal}
            Time: {current_time} (Lagos Time)
            
            Strategy Details:
            - EMA Fast: {ema_fast_length} periods
            - EMA Slow: {ema_slow_length} periods
            - EMA Trend: {ema_trend_length} periods
            - Confirmed with NWE crossover/crossunder
            
            This is an automated message from your trading bot.
            """
            send_email(email_subject, email_body)
        else:
            print(f"{symbol} - No Valid Signal at {current_time}")

    except Exception as e:
        error_msg = f"{symbol} - Error: {str(e)}"
        print(error_msg)
        # Send error notification email
        send_email("Crypto Bot Error", error_msg)

# Check signals for all symbols
print(f"Checking signals at: {datetime.now(lagos_tz).strftime('%Y-%m-%d %H:%M:%S')} (Lagos Time)")
print("----------------------------------")
for symbol in symbols:
    check_valid_signal(symbol)
