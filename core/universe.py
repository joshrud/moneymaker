"""Full stock universe organized by GICS sector (~261 liquid US equities)."""
from __future__ import annotations

UNIVERSE_BY_SECTOR: dict[str, list[str]] = {
    "information_technology": [
        "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "QCOM", "TXN", "INTC", "MU",
        "AMAT", "LRCX", "KLAC", "ADI", "MRVL", "FTNT", "PANW", "CRM", "ORCL",
        "NOW", "ADBE", "INTU", "CDNS", "SNPS", "IBM", "HPQ", "CSCO", "ACN",
        "CTSH", "JNPR", "DELL", "HPE", "KEYS", "ANSS", "EPAM", "MCHP",
    ],
    "communication_services": [
        "META", "GOOGL", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
        "CHTR", "OMC", "IPG", "WBD", "FOXA", "PARA", "EA", "TTWO",
    ],
    "consumer_discretionary": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "ROST", "BKNG",
        "MAR", "HLT", "CMG", "YUM", "DRI", "F", "GM", "ORLY", "AZO", "DG",
        "DLTR", "BBY", "ETSY", "EBAY", "LVS", "MGM", "ABNB", "EXPE",
    ],
    "consumer_staples": [
        "PG", "KO", "PEP", "WMT", "COST", "TGT", "MDLZ", "GIS", "K",
        "HSY", "CLX", "CL", "EL", "CHD", "HRL", "SJM", "MKC", "CPB",
    ],
    "energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "MPC", "PSX", "VLO",
        "HES", "DVN", "OXY", "BKR", "RRC", "AR", "FANG", "MRO", "APA",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "SCHW",
        "AXP", "V", "MA", "BLK", "BX", "APO", "KKR", "SPGI", "MCO", "ICE",
        "CME", "CB", "PGR", "TRV", "AIG", "MMC", "AON", "COF", "AMP", "MSCI",
    ],
    "health_care": [
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "BMY", "AMGN", "GILD",
        "VRTX", "REGN", "CVS", "CI", "ELV", "HCA", "MDT", "ABT", "BSX",
        "SYK", "ISRG", "DXCM", "HOLX", "IDXX", "TMO", "A", "IQV", "ZBH",
        "BAX", "BDX", "UHS", "HUM", "MOH", "CNC",
    ],
    "industrials": [
        "GE", "HON", "MMM", "CAT", "DE", "RTX", "LMT", "NOC", "GD", "BA",
        "UPS", "FDX", "CSX", "NSC", "UNP", "EMR", "ETN", "PH", "ITW",
        "PCAR", "CMI", "DOV", "ROP", "IEX", "EXPD", "FAST", "GWW", "GNRC",
        "XYL", "IR", "LDOS", "SAIC",
    ],
    "materials": [
        "LIN", "APD", "SHW", "FCX", "NEM", "GOLD", "ALB", "CF", "MOS",
        "NUE", "STLD", "CMC", "VMC", "MLM", "RPM", "AA", "EMN", "PPG", "ECL",
    ],
    "real_estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "EXR", "AVB", "EQR", "SPG",
        "O", "VICI", "WELL", "VTR", "MAA", "CPT", "ARE", "WY",
    ],
    "utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ES",
        "WEC", "CMS", "CNP", "ETR", "PPL", "ED", "AES",
    ],
}

# Flat list of all tickers (261 stocks)
UNIVERSE: list[str] = [
    ticker
    for tickers in UNIVERSE_BY_SECTOR.values()
    for ticker in tickers
]

# Reverse map: ticker -> sector name
SECTOR_MAP: dict[str, str] = {
    ticker: sector
    for sector, tickers in UNIVERSE_BY_SECTOR.items()
    for ticker in tickers
}
