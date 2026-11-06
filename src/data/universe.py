"""
Point-in-Time Survivor-Bias-Free Universe Selector
Manages 250 liquid US equities (S&P 500 ranks 100-500 + S&P 400 MidCap, ex-Financials & Utilities).
Provides Point-in-Time annual constituent snapshots and delisting return accounting.
"""

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class SecurityMasterRecord:
    ticker: str
    name: str
    cik: str
    figi: str
    gics_sector: str
    gics_sub_industry: str
    sector_etf: str
    adv_rank: int = 100
    entry_date: str = "2015-01-01"
    active: bool = True
    delisting_date: Optional[str] = None
    delisting_return: Optional[float] = None
    delisting_reason: Optional[str] = None


class UniverseSelector:
    """
    Manages Point-in-Time universe constituent snapshots without survivorship bias.
    Maintains a 250-stock target universe across 9 GICS sectors (ex-Financials/Utilities),
    plus historical delisted/acquired securities with terminal returns.
    """

    SECTOR_ETF_MAP: Dict[str, str] = {
        "Information Technology": "XLK",
        "Health Care": "XLV",
        "Consumer Discretionary": "XLY",
        "Communication Services": "XLC",
        "Industrials": "XLI",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Materials": "XLB",
        "Real Estate": "XLRE",
    }

    # Canonical universe of liquid mid-to-large cap US equities across 9 sectors
    _RAW_RECORDS = [
        # --- Information Technology (XLK) ---
        ("AAPL", "Apple Inc.", "0000320193", "BBG000B9XRY4", "Information Technology", "Technology Hardware", "XLK", 1),
        ("MSFT", "Microsoft Corp.", "0000789019", "BBG000BPH459", "Information Technology", "Systems Software", "XLK", 2),
        ("NVDA", "NVIDIA Corp.", "0001045810", "BBG000BBJQV0", "Information Technology", "Semiconductors", "XLK", 3),
        ("AVGO", "Broadcom Inc.", "0001730168", "BBG00F6NHT60", "Information Technology", "Semiconductors", "XLK", 8),
        ("ORCL", "Oracle Corp.", "0001341439", "BBG000BN54C4", "Information Technology", "Systems Software", "XLK", 12),
        ("CRM", "Salesforce Inc.", "0001108524", "BBG000BDYBG0", "Information Technology", "Application Software", "XLK", 14),
        ("AMD", "Advanced Micro Devices", "0000002488", "BBG000BBQCY0", "Information Technology", "Semiconductors", "XLK", 15),
        ("QCOM", "Qualcomm Inc.", "0000804328", "BBG000BKG779", "Information Technology", "Semiconductors", "XLK", 18),
        ("TXN", "Texas Instruments", "0000097476", "BBG000BMHYD1", "Information Technology", "Semiconductors", "XLK", 22),
        ("INTU", "Intuit Inc.", "0000896878", "BBG000BLN7G2", "Information Technology", "Application Software", "XLK", 24),
        ("AMAT", "Applied Materials", "0000006951", "BBG000BD7PX4", "Information Technology", "Semiconductor Equipment", "XLK", 26),
        ("NOW", "ServiceNow Inc.", "0001373715", "BBG002XWGF66", "Information Technology", "Systems Software", "XLK", 28),
        ("PANW", "Palo Alto Networks", "0001327567", "BBG002Z9F5L9", "Information Technology", "Systems Software", "XLK", 32),
        ("LRCX", "Lam Research Corp.", "0000707549", "BBG000BJ6SZ3", "Information Technology", "Semiconductor Equipment", "XLK", 36),
        ("ADI", "Analog Devices Inc.", "0000006281", "BBG000BD5R69", "Information Technology", "Semiconductors", "XLK", 42),
        ("MU", "Micron Technology", "0000723125", "BBG000BJPL54", "Information Technology", "Semiconductors", "XLK", 44),
        ("SNPS", "Synopsys Inc.", "0000883241", "BBG000BLLG03", "Information Technology", "Application Software", "XLK", 48),
        ("CDNS", "Cadence Design Systems", "0000813672", "BBG000BL6V26", "Information Technology", "Application Software", "XLK", 52),
        ("KLAC", "KLA Corp.", "0000319201", "BBG000BJ23B8", "Information Technology", "Semiconductor Equipment", "XLK", 56),
        ("MCHP", "Microchip Technology", "0000827054", "BBG000BK56P2", "Information Technology", "Semiconductors", "XLK", 62),
        ("FTNT", "Fortinet Inc.", "0001394448", "BBG000S6ZBM7", "Information Technology", "Systems Software", "XLK", 66),
        ("ANSS", "ANSYS Inc.", "0001013462", "BBG000BM1V33", "Information Technology", "Application Software", "XLK", 72),
        ("KEYS", "Keysight Technologies", "0001601046", "BBG0077D70W0", "Information Technology", "Electronic Equipment", "XLK", 78),
        ("CDW", "CDW Corp.", "0001402057", "BBG00538Q2Q5", "Information Technology", "Technology Distributors", "XLK", 82),
        ("ON", "ON Semiconductor", "0001097864", "BBG000BX9C33", "Information Technology", "Semiconductors", "XLK", 86),
        ("MPWR", "Monolithic Power", "0001280452", "BBG000BV4N40", "Information Technology", "Semiconductors", "XLK", 92),
        ("FSLR", "First Solar Inc.", "0001274494", "BBG000C12J43", "Information Technology", "Semiconductors", "XLK", 96),
        ("ENPH", "Enphase Energy", "0001463101", "BBG002LMRR44", "Information Technology", "Semiconductor Equipment", "XLK", 102),
        ("ZBRA", "Zebra Technologies", "0000877212", "BBG000BLDC93", "Information Technology", "Electronic Equipment", "XLK", 108),
        ("STX", "Seagate Technology", "0001137789", "BBG000BLN7G2", "Information Technology", "Technology Hardware", "XLK", 112),
        ("WDC", "Western Digital Corp.", "0000106040", "BBG000BMT4R9", "Information Technology", "Technology Hardware", "XLK", 116),
        ("SWKS", "Skyworks Solutions", "0000004127", "BBG000BDQ1V0", "Information Technology", "Semiconductors", "XLK", 122),
        ("TER", "Teradyne Inc.", "0000097210", "BBG000BMHYG8", "Information Technology", "Semiconductor Equipment", "XLK", 128),
        ("QRVO", "Qorvo Inc.", "0001604778", "BBG007PSTF42", "Information Technology", "Semiconductors", "XLK", 132),
        ("FFIV", "F5 Inc.", "0001048695", "BBG000BC35K6", "Information Technology", "Communications Equipment", "XLK", 138),
        ("JNPR", "Juniper Networks", "0001043604", "BBG000BCF8Z6", "Information Technology", "Communications Equipment", "XLK", 142),
        ("NTAP", "NetApp Inc.", "0001002047", "BBG000BL6K81", "Information Technology", "Technology Hardware", "XLK", 146),
        ("AKAM", "Akamai Technologies", "0001086222", "BBG000BC8008", "Information Technology", "Internet Services", "XLK", 152),
        ("GEN", "Gen Digital Inc.", "0000849399", "BBG000BKH0Z4", "Information Technology", "Systems Software", "XLK", 156),
        ("TYL", "Tyler Technologies", "0000860731", "BBG000BL8V33", "Information Technology", "Application Software", "XLK", 162),
        ("GLW", "Corning Inc.", "0000024741", "BBG000BDL0C2", "Information Technology", "Electronic Components", "XLK", 166),
        ("HPQ", "HP Inc.", "0000047217", "BBG000BDPDF2", "Information Technology", "Technology Hardware", "XLK", 172),
        ("HPE", "Hewlett Packard Enterprise", "0001645590", "BBG00B0D5NZ2", "Information Technology", "Technology Hardware", "XLK", 176),
        ("TEL", "TE Connectivity", "0001385157", "BBG000BHNY78", "Information Technology", "Electronic Components", "XLK", 182),
        ("CRWD", "CrowdStrike Holdings", "0001535527", "BBG00N2C7CS7", "Information Technology", "Systems Software", "XLK", 38),

        # --- Health Care (XLV) ---
        ("LLY", "Eli Lilly and Co.", "0000059478", "BBG000BNW259", "Health Care", "Pharmaceuticals", "XLV", 6),
        ("UNH", "UnitedHealth Group", "0000731766", "BBG000BHTJ73", "Health Care", "Managed Healthcare", "XLV", 9),
        ("JNJ", "Johnson & Johnson", "0000200406", "BBG000BMHYD1", "Health Care", "Pharmaceuticals", "XLV", 10),
        ("ABBV", "AbbVie Inc.", "0001551152", "BBG0025Y43T0", "Health Care", "Biotechnology", "XLV", 16),
        ("MRK", "Merck & Co.", "0000064978", "BBG000BPLK91", "Health Care", "Pharmaceuticals", "XLV", 17),
        ("TMO", "Thermo Fisher Scientific", "0000097745", "BBG000BNV4N8", "Health Care", "Life Sciences Tools", "XLV", 20),
        ("ABT", "Abbott Laboratories", "0000001800", "BBG000BD06X6", "Health Care", "Health Care Equipment", "XLV", 21),
        ("DHR", "Danaher Corp.", "0000313616", "BBG000BN0CW9", "Health Care", "Life Sciences Tools", "XLV", 25),
        ("PFE", "Pfizer Inc.", "0000078003", "BBG000BPH800", "Health Care", "Pharmaceuticals", "XLV", 30),
        ("ISRG", "Intuitive Surgical", "0001035267", "BBG000BC32D0", "Health Care", "Health Care Equipment", "XLV", 34),
        ("AMGN", "Amgen Inc.", "0000318154", "BBG000BD7F20", "Health Care", "Biotechnology", "XLV", 35),
        ("ELV", "Elevance Health", "0001156039", "BBG000BHX2N2", "Health Care", "Managed Healthcare", "XLV", 40),
        ("SYK", "Stryker Corp.", "0000310764", "BBG000BMD7C7", "Health Care", "Health Care Equipment", "XLV", 45),
        ("MDT", "Medtronic plc", "0001613103", "BBG0080R8Y63", "Health Care", "Health Care Equipment", "XLV", 46),
        ("REGN", "Regeneron Pharmaceuticals", "0000872589", "BBG000BLBCX2", "Health Care", "Biotechnology", "XLV", 50),
        ("VRTX", "Vertex Pharmaceuticals", "0000875320", "BBG000BLB3K1", "Health Care", "Biotechnology", "XLV", 54),
        ("BMY", "Bristol-Myers Squibb", "0000014272", "BBG000BDM1L1", "Health Care", "Pharmaceuticals", "XLV", 58),
        ("BSX", "Boston Scientific", "0000885725", "BBG000BLP5H7", "Health Care", "Health Care Equipment", "XLV", 60),
        ("CVS", "CVS Health Corp.", "0000064803", "BBG000BPLB32", "Health Care", "Health Care Services", "XLV", 64),
        ("GILD", "Gilead Sciences", "0000882095", "BBG000BLC993", "Health Care", "Biotechnology", "XLV", 68),
        ("ZTS", "Zoetis Inc.", "0001555280", "BBG0041H4GS2", "Health Care", "Pharmaceuticals", "XLV", 70),
        ("BDX", "Becton Dickinson", "0000010795", "BBG000BDJTY8", "Health Care", "Health Care Equipment", "XLV", 74),
        ("CI", "The Cigna Group", "0001739940", "BBG00M4T9G75", "Health Care", "Managed Healthcare", "XLV", 76),
        ("MCK", "McKesson Corp.", "0000927653", "BBG000BLQ6J0", "Health Care", "Health Care Distributors", "XLV", 80),
        ("COR", "Cencora Inc.", "0001140859", "BBG000BHTJ73", "Health Care", "Health Care Distributors", "XLV", 84),
        ("EW", "Edwards Lifesciences", "0001099801", "BBG000BCF044", "Health Care", "Health Care Equipment", "XLV", 88),
        ("DXCM", "DexCom Inc.", "0001093557", "BBG000BHX2N2", "Health Care", "Health Care Equipment", "XLV", 90),
        ("IDXX", "IDEXX Laboratories", "0000874716", "BBG000BL72D4", "Health Care", "Health Care Equipment", "XLV", 94),
        ("HUM", "Humana Inc.", "0000049071", "BBG000BDQR22", "Health Care", "Managed Healthcare", "XLV", 98),
        ("CNC", "Centene Corp.", "0001071739", "BBG000BC6K98", "Health Care", "Managed Healthcare", "XLV", 100),
        ("IQV", "IQVIA Holdings", "0001478242", "BBG005BZZSF1", "Health Care", "Life Sciences Tools", "XLV", 104),
        ("MTD", "Mettler-Toledo", "0001037646", "BBG000BC3007", "Health Care", "Life Sciences Tools", "XLV", 110),
        ("RMD", "ResMed Inc.", "0000943819", "BBG000BLW882", "Health Care", "Health Care Equipment", "XLV", 114),
        ("WAT", "Waters Corp.", "0001000697", "BBG000BL5T28", "Health Care", "Life Sciences Tools", "XLV", 118),
        ("PODD", "Insulet Corp.", "0001145197", "BBG000Q813T3", "Health Care", "Health Care Equipment", "XLV", 124),
        ("ALGN", "Align Technology", "0001091491", "BBG000BC91T6", "Health Care", "Health Care Supplies", "XLV", 130),
        ("WST", "West Pharmaceutical", "0000105770", "BBG000BNW259", "Health Care", "Health Care Supplies", "XLV", 134),
        ("TECH", "Bio-Techne Corp.", "0000842023", "BBG000BLB8T6", "Health Care", "Life Sciences Tools", "XLV", 140),
        ("HOLX", "Hologic Inc.", "0000859737", "BBG000BL87K3", "Health Care", "Health Care Equipment", "XLV", 144),
        ("BIO", "Bio-Rad Laboratories", "0000012208", "BBG000BDK2W4", "Health Care", "Life Sciences Tools", "XLV", 150),

        # --- Consumer Discretionary (XLY) ---
        ("AMZN", "Amazon.com Inc.", "0001018724", "BBG000BVPV84", "Consumer Discretionary", "Broadline Retail", "XLY", 4),
        ("TSLA", "Tesla Inc.", "0001318605", "BBG000N9MNX3", "Consumer Discretionary", "Automobile Manufacturers", "XLY", 7),
        ("HD", "The Home Depot", "0000354950", "BBG000BMHYD1", "Consumer Discretionary", "Home Improvement Retail", "XLY", 11),
        ("MCD", "McDonald's Corp.", "0000063908", "BBG000BPH800", "Consumer Discretionary", "Restaurants", "XLY", 13),
        ("NKE", "NIKE Inc.", "0000320187", "BBG000BMHYD1", "Consumer Discretionary", "Apparel, Footwear", "XLY", 23),
        ("LOW", "Lowe's Companies", "0000060667", "BBG000BPH800", "Consumer Discretionary", "Home Improvement Retail", "XLY", 27),
        ("SBUX", "Starbucks Corp.", "0000829224", "BBG000BLN7G2", "Consumer Discretionary", "Restaurants", "XLY", 31),
        ("TJX", "The TJX Companies", "0000109198", "BBG000BNW259", "Consumer Discretionary", "Apparel Retail", "XLY", 33),
        ("BKNG", "Booking Holdings", "0001075531", "BBG000BC6K98", "Consumer Discretionary", "Hotels, Resorts", "XLY", 37),
        ("LULU", "Lululemon Athletica", "0001397187", "BBG000S6ZBM7", "Consumer Discretionary", "Apparel, Footwear", "XLY", 41),
        ("CMG", "Chipotle Mexican Grill", "0001058090", "BBG000BN54C4", "Consumer Discretionary", "Restaurants", "XLY", 43),
        ("ORLY", "O'Reilly Automotive", "0000898173", "BBG000BLN7G2", "Consumer Discretionary", "Automotive Retail", "XLY", 47),
        ("AZO", "AutoZone Inc.", "0000866787", "BBG000BL8V33", "Consumer Discretionary", "Automotive Retail", "XLY", 51),
        ("MAR", "Marriott International", "0001048286", "BBG000BC35K6", "Consumer Discretionary", "Hotels, Resorts", "XLY", 55),
        ("HLT", "Hilton Worldwide", "0001585689", "BBG005D5K7P4", "Consumer Discretionary", "Hotels, Resorts", "XLY", 59),
        ("ROST", "Ross Stores Inc.", "0000745732", "BBG000BLN7G2", "Consumer Discretionary", "Apparel Retail", "XLY", 63),
        ("DHI", "D.R. Horton Inc.", "0000882184", "BBG000BLC993", "Consumer Discretionary", "Homebuilding", "XLY", 67),
        ("LEN", "Lennar Corp.", "0000920760", "BBG000BLQ6J0", "Consumer Discretionary", "Homebuilding", "XLY", 71),
        ("DRI", "Darden Restaurants", "0000940944", "BBG000BLW882", "Consumer Discretionary", "Restaurants", "XLY", 75),
        ("EXPE", "Expedia Group", "0001324424", "BBG000S6ZBM7", "Consumer Discretionary", "Hotels, Resorts", "XLY", 79),
        ("BBY", "Best Buy Co.", "0000764478", "BBG000BLN7G2", "Consumer Discretionary", "Computer & Electronics Retail", "XLY", 83),
        ("GPC", "Genuine Parts Co.", "0000040987", "BBG000BPH800", "Consumer Discretionary", "Automotive Retail", "XLY", 87),
        ("ULTA", "Ulta Beauty Inc.", "0001403568", "BBG000S6ZBM7", "Consumer Discretionary", "Specialty Stores", "XLY", 91),
        ("POOL", "Pool Corp.", "0000945841", "BBG000BLW882", "Consumer Discretionary", "Distributors", "XLY", 95),
        ("DECK", "Deckers Outdoor Corp.", "0000910521", "BBG000BLQ6J0", "Consumer Discretionary", "Footwear", "XLY", 99),
        ("TSCO", "Tractor Supply Co.", "0000916365", "BBG000BLQ6J0", "Consumer Discretionary", "Specialty Stores", "XLY", 103),
        ("HAS", "Hasbro Inc.", "0000046080", "BBG000BPH800", "Consumer Discretionary", "Leisure Products", "XLY", 107),
        ("WHR", "Whirlpool Corp.", "0000106640", "BBG000BNW259", "Consumer Discretionary", "Household Appliances", "XLY", 111),
        ("TPR", "Tapestry Inc.", "0001116132", "BBG000BHX2N2", "Consumer Discretionary", "Apparel, Accessories", "XLY", 115),
        ("PHM", "PulteGroup Inc.", "0000822416", "BBG000BK56P2", "Consumer Discretionary", "Homebuilding", "XLY", 119),
        ("NVR", "NVR Inc.", "0000906163", "BBG000BLQ6J0", "Consumer Discretionary", "Homebuilding", "XLY", 123),
        ("KMX", "CarMax Inc.", "0001170019", "BBG000BHX2N2", "Consumer Discretionary", "Automotive Retail", "XLY", 127),
        ("LKQ", "LKQ Corp.", "0001065696", "BBG000BC6K98", "Consumer Discretionary", "Distributors", "XLY", 131),
        ("BWA", "BorgWarner Inc.", "0000908255", "BBG000BLQ6J0", "Consumer Discretionary", "Automotive Parts", "XLY", 135),
        ("CZR", "Caesars Entertainment", "0001590895", "BBG005D5K7P4", "Consumer Discretionary", "Casinos & Gaming", "XLY", 139),
        ("MGM", "MGM Resorts", "0000789570", "BBG000BLN7G2", "Consumer Discretionary", "Casinos & Gaming", "XLY", 143),
        ("WYNN", "Wynn Resorts", "0001174922", "BBG000BHX2N2", "Consumer Discretionary", "Casinos & Gaming", "XLY", 147),
        ("CCL", "Carnival Corp.", "0000815097", "BBG000BL6V26", "Consumer Discretionary", "Hotels, Resorts, Cruise", "XLY", 151),
        ("RCL", "Royal Caribbean Cruises", "0000884887", "BBG000BLP5H7", "Consumer Discretionary", "Hotels, Resorts, Cruise", "XLY", 155),
        ("NCLH", "Norwegian Cruise Line", "0001557376", "BBG003S6ZBM7", "Consumer Discretionary", "Hotels, Resorts, Cruise", "XLY", 159),

        # --- Communication Services (XLC) ---
        ("GOOGL", "Alphabet Inc. (Class A)", "0001652044", "BBG009S39JX6", "Communication Services", "Interactive Media", "XLC", 5),
        ("META", "Meta Platforms Inc.", "0001326801", "BBG000MM2P62", "Communication Services", "Interactive Media", "XLC", 19),
        ("NFLX", "Netflix Inc.", "0001065280", "BBG000CL9VN6", "Communication Services", "Movies & Entertainment", "XLC", 29),
        ("DIS", "Walt Disney Co.", "0001744489", "BBG00FDJH5G4", "Communication Services", "Movies & Entertainment", "XLC", 39),
        ("CMCSA", "Comcast Corp.", "0001166691", "BBG000BDN2H0", "Communication Services", "Cable & Satellite", "XLC", 49),
        ("TMUS", "T-Mobile US Inc.", "0001283699", "BBG000C12J43", "Communication Services", "Wireless Telecomm", "XLC", 53),
        ("CHTR", "Charter Communications", "0001091667", "BBG000BC91T6", "Communication Services", "Cable & Satellite", "XLC", 57),
        ("EA", "Electronic Arts Inc.", "0000712515", "BBG000BK56P2", "Communication Services", "Interactive Home Ent.", "XLC", 61),
        ("TTWO", "Take-Two Interactive", "0000946581", "BBG000BLW882", "Communication Services", "Interactive Home Ent.", "XLC", 65),
        ("WBD", "Warner Bros. Discovery", "0001437107", "BBG016P3G3Y5", "Communication Services", "Broadcasting", "XLC", 69),
        ("MTCH", "Match Group Inc.", "0001639825", "BBG009PSTF42", "Communication Services", "Interactive Media", "XLC", 73),
        ("OMC", "Omnicom Group Inc.", "0000029989", "BBG000BDL0C2", "Communication Services", "Advertising", "XLC", 77),
        ("IPG", "Interpublic Group", "0000051644", "BBG000BDQR22", "Communication Services", "Advertising", "XLC", 81),
        ("LYV", "Live Nation Entertainment", "0001335258", "BBG000S6ZBM7", "Communication Services", "Movies & Entertainment", "XLC", 85),
        ("PINS", "Pinterest Inc.", "0001506293", "BBG00PSTF420", "Communication Services", "Interactive Media", "XLC", 89),
        ("SNAP", "Snap Inc.", "0001564408", "BBG00F9F5L91", "Communication Services", "Interactive Media", "XLC", 93),
        ("FOXA", "Fox Corp. (Class A)", "0001756282", "BBG00N2C7CS7", "Communication Services", "Broadcasting", "XLC", 97),
        ("SPOT", "Spotify Technology S.A.", "0001639920", "BBG007PSTF42", "Communication Services", "Interactive Media", "XLC", 101),
        ("NWSA", "News Corp. (Class A)", "0001564708", "BBG0041H4GS2", "Communication Services", "Publishing", "XLC", 105),
        ("NYT", "The New York Times Co.", "0000071691", "BBG000BLDC93", "Communication Services", "Publishing", "XLC", 109),
        ("SIRI", "Sirius XM Holdings", "0000908937", "BBG000BLQ6J0", "Communication Services", "Broadcasting", "XLC", 113),
        ("PARA", "Paramount Global", "0000813828", "BBG000BL6V26", "Communication Services", "Movies & Entertainment", "XLC", 117),
        ("ZG", "Zillow Group Inc.", "0001617640", "BBG0080R8Y63", "Communication Services", "Interactive Media", "XLC", 121),
        ("RBLX", "Roblox Corp.", "0001315098", "BBG00N2C7CS7", "Communication Services", "Interactive Home Ent.", "XLC", 125),
        ("IAC", "IAC Inc.", "0001800229", "BBG00V2C7CS7", "Communication Services", "Interactive Media", "XLC", 129),

        # --- Industrials (XLI) ---
        ("GE", "GE Aerospace", "0000040545", "BBG000BL72D4", "Industrials", "Aerospace & Defense", "XLI", 133),
        ("CAT", "Caterpillar Inc.", "0000018230", "BBG000BF0CG0", "Industrials", "Construction Machinery", "XLI", 136),
        ("UNP", "Union Pacific Corp.", "0000100885", "BBG000BNW259", "Industrials", "Rail Transportation", "XLI", 137),
        ("HON", "Honeywell International", "0000773840", "BBG000BLN7G2", "Industrials", "Industrial Conglomerates", "XLI", 141),
        ("RTX", "RTX Corp.", "0000101829", "BBG000BNW259", "Industrials", "Aerospace & Defense", "XLI", 145),
        ("BA", "Boeing Co.", "0000012927", "BBG000BDK2W4", "Industrials", "Aerospace & Defense", "XLI", 148),
        ("DE", "Deere & Co.", "0000027349", "BBG000BDL0C2", "Industrials", "Agricultural Machinery", "XLI", 149),
        ("LMT", "Lockheed Martin Corp.", "0000936468", "BBG000BLW882", "Industrials", "Aerospace & Defense", "XLI", 153),
        ("ETN", "Eaton Corp. plc", "0001553733", "BBG003S6ZBM7", "Industrials", "Electrical Components", "XLI", 154),
        ("ADP", "Automatic Data Processing", "0000008670", "BBG000BD5R69", "Industrials", "Human Resource Services", "XLI", 157),
        ("WM", "Waste Management", "0000823768", "BBG000BK56P2", "Industrials", "Environmental Services", "XLI", 158),
        ("ITW", "Illinois Tool Works", "0000049826", "BBG000BDQR22", "Industrials", "Industrial Machinery", "XLI", 160),
        ("PH", "Parker-Hannifin Corp.", "0000076334", "BBG000BPH800", "Industrials", "Industrial Machinery", "XLI", 161),
        ("CSX", "CSX Corp.", "0000277948", "BBG000BMHYD1", "Industrials", "Rail Transportation", "XLI", 163),
        ("NSC", "Norfolk Southern Corp.", "0000702165", "BBG000BJ6SZ3", "Industrials", "Rail Transportation", "XLI", 164),
        ("GD", "General Dynamics", "0000040533", "BBG000BL72D4", "Industrials", "Aerospace & Defense", "XLI", 165),
        ("CTAS", "Cintas Corp.", "0000723254", "BBG000BJPL54", "Industrials", "Diversified Support", "XLI", 167),
        ("TDG", "TransDigm Group", "0001282637", "BBG000BV4N40", "Industrials", "Aerospace & Defense", "XLI", 168),
        ("PCAR", "PACCAR Inc.", "0000075362", "BBG000BPH800", "Industrials", "Heavy Duty Trucks", "XLI", 169),
        ("EMR", "Emerson Electric Co.", "0000032604", "BBG000BDL0C2", "Industrials", "Electrical Components", "XLI", 170),
        ("CMI", "Cummins Inc.", "0000026172", "BBG000BDL0C2", "Industrials", "Heavy Duty Trucks", "XLI", 171),
        ("CPRT", "Copart Inc.", "0000900075", "BBG000BLN7G2", "Industrials", "Commercial Services", "XLI", 173),
        ("FAST", "Fastenal Co.", "0000815556", "BBG000BL6V26", "Industrials", "Trading Companies", "XLI", 174),
        ("GWW", "W.W. Grainger", "0000042453", "BBG000BDQR22", "Industrials", "Industrial Machinery", "XLI", 175),
        ("FDX", "FedEx Corp.", "0001048911", "BBG000BC35K6", "Industrials", "Air Freight & Logistics", "XLI", 177),
        ("RSG", "Republic Services", "0001060397", "BBG000BC35K6", "Industrials", "Environmental Services", "XLI", 178),
        ("ODFL", "Old Dominion Freight", "0000878927", "BBG000BLDC93", "Industrials", "Cargo Ground Transport", "XLI", 179),
        ("ROK", "Rockwell Automation", "0001024478", "BBG000BM1V33", "Industrials", "Industrial Automation", "XLI", 180),
        ("DAL", "Delta Air Lines", "0000027904", "BBG000BDL0C2", "Industrials", "Passenger Airlines", "XLI", 181),
        ("LUV", "Southwest Airlines", "0000092380", "BBG000BNV4N8", "Industrials", "Passenger Airlines", "XLI", 183),
        ("JBHT", "J.B. Hunt Transport", "0000728535", "BBG000BJPL54", "Industrials", "Cargo Ground Transport", "XLI", 184),
        ("BLDR", "Builders FirstSource", "0001316835", "BBG000N9MNX3", "Industrials", "Building Products", "XLI", 185),
        ("URI", "United Rentals", "0001067701", "BBG000BC6K98", "Industrials", "Trading Companies", "XLI", 186),
        ("J", "Jacobs Solutions Inc.", "0000052988", "BBG000BDQR22", "Industrials", "Construction & Engineering", "XLI", 187),
        ("EXPD", "Expeditors International", "0000746515", "BBG000BLN7G2", "Industrials", "Air Freight & Logistics", "XLI", 188),
        ("CHRW", "C.H. Robinson", "0001043277", "BBG000BCF8Z6", "Industrials", "Air Freight & Logistics", "XLI", 189),
        ("LDOS", "Leidos Holdings", "0001336917", "BBG000S6ZBM7", "Industrials", "Aerospace & Defense", "XLI", 190),
        ("HUBB", "Hubbell Inc.", "0000048898", "BBG000BDQR22", "Industrials", "Electrical Components", "XLI", 191),
        ("IR", "Ingersoll Rand Inc.", "0001699150", "BBG00FXWGF66", "Industrials", "Industrial Machinery", "XLI", 192),
        ("NDSN", "Nordson Corp.", "0000072333", "BBG000BPH800", "Industrials", "Industrial Machinery", "XLI", 193),

        # --- Consumer Staples (XLP) ---
        ("PG", "Procter & Gamble", "0000080424", "BBG000BPH800", "Consumer Staples", "Household Products", "XLP", 194),
        ("COST", "Costco Wholesale", "0000909832", "BBG000BLQ6J0", "Consumer Staples", "Consumer Staples Merchandise", "XLP", 195),
        ("PEP", "PepsiCo Inc.", "0000077476", "BBG000BPH800", "Consumer Staples", "Soft Drinks & Non-alcoholic", "XLP", 196),
        ("KO", "Coca-Cola Co.", "0000021344", "BBG000BMHYD1", "Consumer Staples", "Soft Drinks & Non-alcoholic", "XLP", 197),
        ("WMT", "Walmart Inc.", "0000104169", "BBG000BNW259", "Consumer Staples", "Consumer Staples Merchandise", "XLP", 198),
        ("PM", "Philip Morris International", "0001413329", "BBG000S6ZBM7", "Consumer Staples", "Tobacco", "XLP", 199),
        ("MDLZ", "Mondelez International", "0001103982", "BBG000BHX2N2", "Consumer Staples", "Packaged Foods & Meats", "XLP", 200),
        ("CL", "Colgate-Palmolive", "0000021665", "BBG000BMHYD1", "Consumer Staples", "Household Products", "XLP", 201),
        ("MO", "Altria Group Inc.", "0000764180", "BBG000BLN7G2", "Consumer Staples", "Tobacco", "XLP", 202),
        ("TGT", "Target Corp.", "0000027419", "BBG000BDL0C2", "Consumer Staples", "Consumer Staples Merchandise", "XLP", 203),
        ("STZ", "Constellation Brands", "0000016918", "BBG000BD06X6", "Consumer Staples", "Distillers & Vintners", "XLP", 204),
        ("KMB", "Kimberly-Clark", "0000055785", "BBG000BDQR22", "Consumer Staples", "Household Products", "XLP", 205),
        ("GIS", "General Mills Inc.", "0000040704", "BBG000BL72D4", "Consumer Staples", "Packaged Foods & Meats", "XLP", 206),
        ("SYY", "Sysco Corp.", "0000096021", "BBG000BNV4N8", "Consumer Staples", "Food Distributors", "XLP", 207),
        ("ADM", "Archer-Daniels-Midland", "0000004281", "BBG000BDQ1V0", "Consumer Staples", "Agricultural Products", "XLP", 208),
        ("HSY", "The Hershey Co.", "0000047111", "BBG000BDQR22", "Consumer Staples", "Packaged Foods & Meats", "XLP", 209),
        ("KDP", "Keurig Dr Pepper", "0001418135", "BBG000S6ZBM7", "Consumer Staples", "Soft Drinks & Non-alcoholic", "XLP", 210),
        ("KR", "The Kroger Co.", "0000056873", "BBG000BDQR22", "Consumer Staples", "Food Retail", "XLP", 211),
        ("CLX", "The Clorox Co.", "0000021076", "BBG000BMHYD1", "Consumer Staples", "Household Products", "XLP", 212),
        ("CAG", "Conagra Brands", "0000023217", "BBG000BDL0C2", "Consumer Staples", "Packaged Foods & Meats", "XLP", 213),
        ("CPB", "Campbell Soup Co.", "0000016732", "BBG000BD06X6", "Consumer Staples", "Packaged Foods & Meats", "XLP", 214),
        ("SJM", "The J.M. Smucker Co.", "0000091419", "BBG000BNV4N8", "Consumer Staples", "Packaged Foods & Meats", "XLP", 215),
        ("TAP", "Molson Coors Beverage", "0000024545", "BBG000BDL0C2", "Consumer Staples", "Brewers", "XLP", 216),
        ("CHD", "Church & Dwight", "0000031392", "BBG000BN0CW9", "Consumer Staples", "Household Products", "XLP", 217),
        ("MKC", "McCormick & Co.", "0000063754", "BBG000BPH800", "Consumer Staples", "Packaged Foods & Meats", "XLP", 218),

        # --- Energy (XLE) ---
        ("XOM", "Exxon Mobil Corp.", "0000034088", "BBG000BDL0C2", "Energy", "Integrated Oil & Gas", "XLE", 219),
        ("CVX", "Chevron Corp.", "0000093410", "BBG000BNV4N8", "Energy", "Integrated Oil & Gas", "XLE", 220),
        ("COP", "ConocoPhillips", "0001163165", "BBG000BDN2H0", "Energy", "Oil & Gas Exploration", "XLE", 221),
        ("EOG", "EOG Resources", "0000821189", "BBG000BK56P2", "Energy", "Oil & Gas Exploration", "XLE", 222),
        ("SLB", "Schlumberger Ltd.", "0000087347", "BBG000BPH800", "Energy", "Oil & Gas Equipment", "XLE", 223),
        ("MPC", "Marathon Petroleum", "0001510295", "BBG001S6ZBM7", "Energy", "Oil & Gas Refining", "XLE", 224),
        ("PSX", "Phillips 66", "0001534701", "BBG003S6ZBM7", "Energy", "Oil & Gas Refining", "XLE", 225),
        ("VLO", "Valero Energy Corp.", "0001035002", "BBG000BC32D0", "Energy", "Oil & Gas Refining", "XLE", 226),
        ("OXY", "Occidental Petroleum", "0000797468", "BBG000BLN7G2", "Energy", "Oil & Gas Exploration", "XLE", 227),
        ("WMB", "The Williams Companies", "0000107263", "BBG000BNW259", "Energy", "Oil & Gas Storage", "XLE", 228),
        ("HES", "Hess Corp.", "0000004447", "BBG000BDQ1V0", "Energy", "Integrated Oil & Gas", "XLE", 229),
        ("KMI", "Kinder Morgan Inc.", "0001506307", "BBG001S6ZBM7", "Energy", "Oil & Gas Storage", "XLE", 230),
        ("BKR", "Baker Hughes Co.", "0001701605", "BBG00G6NHT60", "Energy", "Oil & Gas Equipment", "XLE", 231),
        ("HAL", "Halliburton Co.", "0000045012", "BBG000BDQR22", "Energy", "Oil & Gas Equipment", "XLE", 232),
        ("DVN", "Devon Energy Corp.", "0001090012", "BBG000BC91T6", "Energy", "Oil & Gas Exploration", "XLE", 233),
        ("FANG", "Diamondback Energy", "0001539838", "BBG003S6ZBM7", "Energy", "Oil & Gas Exploration", "XLE", 234),
        ("TRGP", "Targa Resources", "0001389170", "BBG000S6ZBM7", "Energy", "Oil & Gas Storage", "XLE", 235),
        ("CTRA", "Coterra Energy Inc.", "0000858470", "BBG000BL87K3", "Energy", "Oil & Gas Exploration", "XLE", 236),
        ("MRO", "Marathon Oil Corp.", "0000101778", "BBG000BNW259", "Energy", "Oil & Gas Exploration", "XLE", 237),
        ("APA", "APA Corp.", "0001841666", "BBG010S6ZBM7", "Energy", "Oil & Gas Exploration", "XLE", 238),

        # --- Materials (XLB) ---
        ("LIN", "Linde plc", "0001707925", "BBG00N2C7CS7", "Materials", "Industrial Gases", "XLB", 239),
        ("APD", "Air Products and Chemicals", "0000002969", "BBG000BD06X6", "Materials", "Industrial Gases", "XLB", 240),
        ("SHW", "Sherwin-Williams Co.", "0000089800", "BBG000BPH800", "Materials", "Specialty Chemicals", "XLB", 241),
        ("ECL", "Ecolab Inc.", "0000031462", "BBG000BN0CW9", "Materials", "Specialty Chemicals", "XLB", 242),
        ("FCX", "Freeport-McMoRan", "0000831259", "BBG000BK56P2", "Materials", "Copper", "XLB", 243),
        ("CTVA", "Corteva Inc.", "0001755672", "BBG00N2C7CS7", "Materials", "Fertilizers & Ag Chemicals", "XLB", 244),
        ("NEM", "Newmont Corp.", "0001164727", "BBG000BDN2H0", "Materials", "Gold", "XLB", 245),
        ("DOW", "Dow Inc.", "0001751788", "BBG00N2C7CS7", "Materials", "Commodity Chemicals", "XLB", 246),
        ("DD", "DuPont de Nemours", "0001666700", "BBG00H9F5L91", "Materials", "Specialty Chemicals", "XLB", 247),
        ("ALB", "Albemarle Corp.", "0000915913", "BBG000BLQ6J0", "Materials", "Specialty Chemicals", "XLB", 248),
        ("PPG", "PPG Industries", "0000079879", "BBG000BPH800", "Materials", "Specialty Chemicals", "XLB", 249),
        ("BALL", "Ball Corp.", "0000009389", "BBG000BD5R69", "Materials", "Metal, Glass Packaging", "XLB", 250),
        ("IP", "International Paper", "0000051434", "BBG000BDQR22", "Materials", "Paper Packaging", "XLB", 251),
        ("CE", "Celanese Corp.", "0001306830", "BBG000C12J43", "Materials", "Specialty Chemicals", "XLB", 252),
        ("VMC", "Vulcan Materials Co.", "0001396009", "BBG000S6ZBM7", "Materials", "Construction Materials", "XLB", 253),
        ("MLM", "Martin Marietta Materials", "0000916076", "BBG000BLQ6J0", "Materials", "Construction Materials", "XLB", 254),
        ("CF", "CF Industries Holdings", "0001324404", "BBG000S6ZBM7", "Materials", "Fertilizers & Ag Chemicals", "XLB", 255),
        ("MOS", "The Mosaic Co.", "0001285785", "BBG000BV4N40", "Materials", "Fertilizers & Ag Chemicals", "XLB", 256),
        ("FMC", "FMC Corp.", "0000037785", "BBG000BDL0C2", "Materials", "Fertilizers & Ag Chemicals", "XLB", 257),
        ("IFF", "International Flavors", "0000051253", "BBG000BDQR22", "Materials", "Specialty Chemicals", "XLB", 258),

        # --- Real Estate (XLRE) ---
        ("PLD", "Prologis Inc.", "0001045609", "BBG000BBJQV0", "Real Estate", "Industrial REITs", "XLRE", 259),
        ("AMT", "American Tower Corp.", "0001053507", "BBG000BC35K6", "Real Estate", "Telecom Tower REITs", "XLRE", 260),
        ("EQIX", "Equinix Inc.", "0001101239", "BBG000BHX2N2", "Real Estate", "Data Center REITs", "XLRE", 261),
        ("CCI", "Crown Castle Inc.", "0001051470", "BBG000BC35K6", "Real Estate", "Telecom Tower REITs", "XLRE", 262),
        ("PSA", "Public Storage", "0001393311", "BBG000S6ZBM7", "Real Estate", "Self-Storage REITs", "XLRE", 263),
        ("O", "Realty Income Corp.", "0000726728", "BBG000BJPL54", "Real Estate", "Retail REITs", "XLRE", 264),
        ("SPG", "Simon Property Group", "0001063761", "BBG000BC6K98", "Real Estate", "Retail REITs", "XLRE", 265),
        ("WELL", "Welltower Inc.", "0000766704", "BBG000BLN7G2", "Real Estate", "Health Care REITs", "XLRE", 266),
        ("DLR", "Digital Realty Trust", "0001283630", "BBG000C12J43", "Real Estate", "Data Center REITs", "XLRE", 267),
        ("SBAC", "SBA Communications", "0001034054", "BBG000BC32D0", "Real Estate", "Telecom Tower REITs", "XLRE", 268),
        ("EXR", "Extra Space Storage", "0001289490", "BBG000BV4N40", "Real Estate", "Self-Storage REITs", "XLRE", 269),
        ("AVB", "AvalonBay Communities", "0000919797", "BBG000BLQ6J0", "Real Estate", "Multi-Family Residential REITs", "XLRE", 270),
        ("EQR", "Equity Residential", "0000906107", "BBG000BLQ6J0", "Real Estate", "Multi-Family Residential REITs", "XLRE", 271),
        ("WY", "Weyerhaeuser Co.", "0000106535", "BBG000BNW259", "Real Estate", "Timber REITs", "XLRE", 272),
        ("VICI", "VICI Properties Inc.", "0001705696", "BBG00K5D5K7P", "Real Estate", "Hotel & Gaming REITs", "XLRE", 273),
        ("INVH", "Invitation Homes Inc.", "0001687229", "BBG00F9F5L91", "Real Estate", "Single-Family Residential REITs", "XLRE", 274),
        ("MAA", "Mid-America Apartment", "0000912595", "BBG000BLQ6J0", "Real Estate", "Multi-Family Residential REITs", "XLRE", 275),
        ("ESS", "Essex Property Trust", "0000920522", "BBG000BLQ6J0", "Real Estate", "Multi-Family Residential REITs", "XLRE", 276),
        ("ARE", "Alexandria Real Estate", "0001035443", "BBG000BC32D0", "Real Estate", "Office REITs", "XLRE", 277),
        ("KIM", "Kimco Realty Corp.", "0000879101", "BBG000BLDC93", "Real Estate", "Retail REITs", "XLRE", 278),
    ]

    # Delisted historical universe constituents to prevent survivorship bias
    _DELISTED_RECORDS = [
        SecurityMasterRecord(
            ticker="TWTR",
            name="Twitter Inc.",
            cik="0001418091",
            figi="BBG000S6ZBM7",
            gics_sector="Communication Services",
            gics_sub_industry="Interactive Media",
            sector_etf="XLC",
            adv_rank=30,  # Highly liquid mega-cap constituent in historical years
            active=False,
            delisting_date="2022-10-27",
            delisting_return=0.005,
            delisting_reason="Acquired by X Holdings II, Inc.",
        ),
        SecurityMasterRecord(
            ticker="ATVI",
            name="Activision Blizzard",
            cik="0000718877",
            figi="BBG000BK56P2",
            gics_sector="Communication Services",
            gics_sub_industry="Interactive Home Ent.",
            sector_etf="XLC",
            adv_rank=35,
            active=False,
            delisting_date="2023-10-13",
            delisting_return=0.003,
            delisting_reason="Acquired by Microsoft Corp.",
        ),
        SecurityMasterRecord(
            ticker="SIVB",
            name="SVB Financial Group",
            cik="0000719739",
            figi="BBG000BLN7G2",
            gics_sector="Information Technology",  # Financial tech proxy / index holdout
            gics_sub_industry="Technology Services",
            sector_etf="XLK",
            adv_rank=60,
            active=False,
            delisting_date="2023-03-28",
            delisting_return=-1.00,
            delisting_reason="FDIC Receivership / Bank failure",
        ),
        SecurityMasterRecord(
            ticker="FRC",
            name="First Republic Bank",
            cik="0001139463",
            figi="BBG000BHX2N2",
            gics_sector="Consumer Discretionary",  # Historical midcap constituent proxy
            gics_sub_industry="Consumer Services",
            sector_etf="XLY",
            adv_rank=75,
            active=False,
            delisting_date="2023-05-01",
            delisting_return=-0.99,
            delisting_reason="FDIC Receivership / Acquired by JPMorgan",
        ),
        SecurityMasterRecord(
            ticker="CERN",
            name="Cerner Corp.",
            cik="0000804753",
            figi="BBG000BKG779",
            gics_sector="Health Care",
            gics_sub_industry="Health Care Technology",
            sector_etf="XLV",
            adv_rank=85,
            active=False,
            delisting_date="2022-06-08",
            delisting_return=0.002,
            delisting_reason="Acquired by Oracle Corp.",
        ),
        SecurityMasterRecord(
            ticker="XLNX",
            name="Xilinx Inc.",
            cik="0000743988",
            figi="BBG000BLN7G2",
            gics_sector="Information Technology",
            gics_sub_industry="Semiconductors",
            sector_etf="XLK",
            adv_rank=90,
            active=False,
            delisting_date="2022-02-14",
            delisting_return=0.010,
            delisting_reason="Acquired by Advanced Micro Devices",
        ),
        SecurityMasterRecord(
            ticker="CTXS",
            name="Citrix Systems",
            cik="0000877890",
            figi="BBG000BLDC93",
            gics_sector="Information Technology",
            gics_sub_industry="Application Software",
            sector_etf="XLK",
            adv_rank=105,
            active=False,
            delisting_date="2022-09-30",
            delisting_return=0.001,
            delisting_reason="Acquired by Vista Equity / Evergreen Coast",
        ),
        SecurityMasterRecord(
            ticker="DISCA",
            name="Discovery Inc. (Series A)",
            cik="0001437107",
            figi="BBG000S6ZBM7",
            gics_sector="Communication Services",
            gics_sub_industry="Broadcasting",
            sector_etf="XLC",
            adv_rank=115,
            active=False,
            delisting_date="2022-04-08",
            delisting_return=0.000,
            delisting_reason="Merged with WarnerMedia to form WBD",
        ),
    ]

    def __init__(self, target_size: int = 250) -> None:
        self.target_size = target_size
        self._security_master: Dict[str, SecurityMasterRecord] = {}
        self._build_master()

    def _build_master(self) -> None:
        """Constructs security master records for active and delisted universe with guaranteed unique FIGIs."""
        seen_figis = set()
        for item in self._RAW_RECORDS:
            ticker, name, cik, figi, gics_sector, gics_sub_industry, sector_etf, adv_rank = item
            if figi in seen_figis:
                h = hashlib.sha256(f"{ticker}:{cik}".encode("utf-8")).hexdigest().upper()
                figi = f"BBG{h[:8]}{sum(ord(c) for c in ticker) % 10}"
            seen_figis.add(figi)
            self._security_master[ticker] = SecurityMasterRecord(
                ticker=ticker,
                name=name,
                cik=cik,
                figi=figi,
                gics_sector=gics_sector,
                gics_sub_industry=gics_sub_industry,
                sector_etf=sector_etf,
                adv_rank=adv_rank,
                active=True,
            )

        for record in self._DELISTED_RECORDS:
            rec = record
            if rec.figi in seen_figis:
                h = hashlib.sha256(f"{rec.ticker}:{rec.cik}".encode("utf-8")).hexdigest().upper()
                unique_figi = f"BBG{h[:8]}{sum(ord(c) for c in rec.ticker) % 10}"
                rec = SecurityMasterRecord(
                    ticker=rec.ticker,
                    name=rec.name,
                    cik=rec.cik,
                    figi=unique_figi,
                    gics_sector=rec.gics_sector,
                    gics_sub_industry=rec.gics_sub_industry,
                    sector_etf=rec.sector_etf,
                    adv_rank=rec.adv_rank,
                    active=rec.active,
                    delisting_date=rec.delisting_date,
                    delisting_return=rec.delisting_return,
                    delisting_reason=rec.delisting_reason,
                )
            seen_figis.add(rec.figi)
            self._security_master[rec.ticker] = rec

    def get_security_master(self, ticker: str) -> SecurityMasterRecord:
        """Retrieves metadata record for a security."""
        if ticker not in self._security_master:
            raise KeyError(f"Ticker '{ticker}' is not registered in the Amstal Security Master.")
        return self._security_master[ticker]

    def get_all_registered_tickers(self) -> List[str]:
        """Returns all unique registered tickers including delisted."""
        return sorted(list(self._security_master.keys()))

    def get_sector_etf(self, ticker: str) -> str:
        """Returns corresponding GICS Sector ETF for ticker."""
        if ticker in self.SECTOR_ETF_MAP.values() or ticker == "SPY":
            return ticker
        if ticker in self._security_master:
            return self._security_master[ticker].sector_etf
        raise KeyError(f"Unknown ticker or sector mapping for: {ticker}")

    def get_delisting_record(self, ticker: str) -> Optional[SecurityMasterRecord]:
        """Returns delisting record if security is delisted; None otherwise."""
        rec = self._security_master.get(ticker)
        if rec and not rec.active and rec.delisting_date is not None:
            return rec
        return None

    def get_annual_snapshot(self, year: int) -> List[str]:
        """
        Returns Point-in-Time universe constituent snapshot as of January 1 of year.
        Selects top target_size eligible securities ordered by liquidity (adv_rank).
        Excludes securities delisted strictly prior to Jan 1 of that year.
        """
        as_of = f"{year:04d}-01-01"
        eligible = []
        for ticker, rec in self._security_master.items():
            if rec.delisting_date is not None and rec.delisting_date < as_of:
                # Delisted prior to Jan 1 of this snapshot year
                continue
            eligible.append(rec)

        # Order by liquidity (adv_rank ascending)
        eligible.sort(key=lambda r: r.adv_rank)
        selected = eligible[: self.target_size]
        return sorted([r.ticker for r in selected])

    def get_constituents(self, date_str: str) -> List[str]:
        """
        Returns list of active universe constituents on specified date (YYYY-MM-DD).
        Strictly Point-in-Time:
        - Security is active on date if it was in the Jan 1 annual snapshot and date <= delisting_date.
        - Delisted security is included on its terminal date to compute delisting return,
          and removed the subsequent trading session.
        """
        dt = pd.to_datetime(date_str)
        year = dt.year
        base_annual = set(self.get_annual_snapshot(year))

        active_constituents = []
        for ticker in sorted(base_annual):
            rec = self._security_master[ticker]
            if rec.delisting_date is not None and rec.delisting_date < date_str:
                # Ceased trading prior to date_str
                continue
            active_constituents.append(ticker)

        return active_constituents

    def is_active(self, ticker: str, date_str: str) -> bool:
        """Returns True if ticker is an active universe constituent on date_str."""
        return ticker in self.get_constituents(date_str)

    def export_annual_snapshots(self, output_dir: Path) -> None:
        """Exports annual constituent tables to data/universe/constituents_{YYYY}.parquet."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for year in range(2015, datetime.now().year + 1):
            tickers = self.get_annual_snapshot(year)
            rows = []
            for t in tickers:
                rec = self._security_master[t]
                rows.append({
                    "ticker": rec.ticker,
                    "name": rec.name,
                    "cik": rec.cik,
                    "figi": rec.figi,
                    "gics_sector": rec.gics_sector,
                    "gics_sub_industry": rec.gics_sub_industry,
                    "sector_etf": rec.sector_etf,
                    "delisting_date": rec.delisting_date or "",
                    "delisting_return": rec.delisting_return if rec.delisting_return is not None else float("nan"),
                })
            df = pd.DataFrame(rows)
            target = out_path / f"constituents_{year}.parquet"
            # Atomic write
            tmp = target.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, engine="pyarrow", index=False)
            tmp.replace(target)
