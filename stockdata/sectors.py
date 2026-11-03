"""SIC code -> 11 GICS-style sectors.

SIC is the SEC's (old, industry-of-production) code, current value only, so
this is an approximation of GICS. A short list of mega-caps whose SIC clearly
contradicts their GICS sector is overridden by CIK (see OVERRIDES).
"""
from __future__ import annotations

SECTORS = ("Energy", "Materials", "Industrials", "Consumer Discretionary", "Consumer Staples",
           "Health Care", "Financials", "Information Technology", "Communication Services",
           "Utilities", "Real Estate")

# (low, high, sector) checked in order; first match wins. Specific ranges first.
_RANGES = [
    # --- specific carve-outs
    (1311, 1389, "Energy"),                 # oil & gas extraction, oilfield services
    (1220, 1241, "Energy"),                 # coal
    (1520, 1531, "Consumer Discretionary"),  # homebuilders
    (2080, 2087, "Consumer Staples"),       # beverages
    (2830, 2836, "Health Care"),            # pharma, biotech
    (2840, 2844, "Consumer Staples"),       # soaps, cosmetics
    (2911, 2999, "Energy"),                 # refining
    (3011, 3011, "Consumer Discretionary"),  # tires
    (2451, 2452, "Consumer Discretionary"),  # mobile homes
    (2711, 2741, "Communication Services"),  # newspapers, periodicals, books, publishing
    (3570, 3579, "Information Technology"),  # computers & office equipment
    (3630, 3639, "Consumer Discretionary"),  # household appliances
    (3651, 3652, "Consumer Discretionary"),  # household audio/video
    (3661, 3679, "Information Technology"),  # communications equipment, semiconductors, components
    (3577, 3577, "Information Technology"),
    (3695, 3695, "Information Technology"),  # magnetic & optical media
    (3711, 3716, "Consumer Discretionary"),  # motor vehicles
    (3714, 3714, "Consumer Discretionary"),  # auto parts
    (3751, 3751, "Consumer Discretionary"),  # motorcycles, bicycles
    (3790, 3799, "Consumer Discretionary"),  # misc transportation equipment (RVs)
    (3812, 3812, "Industrials"),            # defense electronics, navigation
    (3820, 3825, "Information Technology"),  # measuring & controlling instruments
    (3826, 3826, "Health Care"),            # laboratory analytical instruments
    (3827, 3829, "Information Technology"),
    (3841, 3851, "Health Care"),            # medical devices, dental, ophthalmic
    (3861, 3861, "Information Technology"),  # photographic equipment
    (3873, 3873, "Consumer Discretionary"),  # watches
    (3910, 3999, "Consumer Discretionary"),  # jewelry, toys, sporting goods, misc
    (4610, 4619, "Energy"),                 # pipelines (ex natural gas)
    (4922, 4923, "Energy"),                 # natural gas transmission
    (4724, 4725, "Consumer Discretionary"),  # travel agencies
    (4950, 4959, "Industrials"),            # waste management
    (4955, 4955, "Industrials"),
    (5045, 5045, "Information Technology"),  # computer wholesale
    (5047, 5047, "Health Care"),            # medical wholesale
    (5065, 5065, "Information Technology"),  # electronic parts wholesale
    (5122, 5122, "Health Care"),            # drug wholesale
    (5140, 5149, "Consumer Staples"),       # grocery wholesale
    (5171, 5172, "Energy"),                 # petroleum wholesale
    (5180, 5182, "Consumer Staples"),       # beer/wine wholesale
    (5190, 5199, "Industrials"),
    (5331, 5331, "Consumer Staples"),       # variety stores (WMT, TGT, DG, DLTR, COST: GICS staples since 2023)
    (5399, 5399, "Consumer Staples"),       # misc general merchandise (warehouse clubs)
    (5400, 5499, "Consumer Staples"),       # food stores
    (5912, 5912, "Consumer Staples"),       # drug stores
    (6500, 6553, "Real Estate"),
    (6798, 6798, "Real Estate"),            # REITs
    (7370, 7379, "Information Technology"),  # software, IT services
    (7389, 7389, "Industrials"),            # business services NEC
    (7510, 7519, "Industrials"),            # car/truck rental
    (7800, 7849, "Communication Services"),  # motion pictures, video rental (NFLX)
    (7812, 7812, "Communication Services"),
    (8000, 8099, "Health Care"),
    (8731, 8734, "Health Care"),            # commercial research labs (CROs)
    (8200, 8299, "Consumer Discretionary"),  # education
    (8300, 8399, "Health Care"),            # social services
    # --- broad divisions
    (800, 899, "Materials"),                # forestry
    (100, 999, "Consumer Staples"),         # agriculture, fishing
    (1000, 1499, "Materials"),              # metal & nonmetal mining
    (1500, 1799, "Industrials"),            # construction
    (2000, 2199, "Consumer Staples"),       # food, tobacco
    (2200, 2399, "Consumer Discretionary"),  # textiles, apparel
    (2400, 2499, "Materials"),              # lumber
    (2500, 2599, "Consumer Discretionary"),  # furniture
    (2600, 2699, "Materials"),              # paper
    (2700, 2799, "Industrials"),            # commercial printing
    (2800, 2899, "Materials"),              # chemicals
    (3000, 3099, "Materials"),              # rubber & plastics
    (3100, 3199, "Consumer Discretionary"),  # leather, footwear
    (3200, 3399, "Materials"),              # glass, cement, primary metals
    (3400, 3569, "Industrials"),            # fabricated metal, machinery
    (3580, 3629, "Industrials"),            # HVAC, electrical equipment
    (3640, 3699, "Industrials"),
    (3700, 3799, "Industrials"),            # aerospace, ships, rail equipment
    (3800, 3899, "Information Technology"),
    (4000, 4799, "Industrials"),            # transportation
    (4800, 4899, "Communication Services"),  # telecom, broadcasting, cable
    (4900, 4999, "Utilities"),
    (5000, 5199, "Industrials"),            # wholesale / distributors
    (5200, 5999, "Consumer Discretionary"),  # retail, restaurants
    (6000, 6499, "Financials"),             # banks, brokers, insurance
    (6700, 6799, "Financials"),             # holding & investment offices, SPACs
    (7000, 7299, "Consumer Discretionary"),  # hotels, personal services
    (7300, 7399, "Industrials"),            # business services
    (7500, 7599, "Consumer Discretionary"),  # auto services
    (7600, 7699, "Industrials"),            # repair services
    (7900, 7999, "Consumer Discretionary"),  # amusement & recreation, casinos
    (8100, 8199, "Industrials"),            # legal
    (8700, 8799, "Industrials"),            # engineering, accounting, consulting
    (8900, 8999, "Industrials"),
]

# CIK -> sector where SIC is clearly misleading for a very large company.
OVERRIDES = {
    1652044: "Communication Services",  # Alphabet (SIC 7370)
    1326801: "Communication Services",  # Meta (SIC 7370)
    1403161: "Financials",              # Visa (SIC 7389), GICS Financials since 2023
    1141391: "Financials",              # Mastercard (SIC 7389)
    1633917: "Financials",              # PayPal (SIC 7389)
    798354: "Financials",               # Fiserv (SIC 7374)
    1136893: "Financials",              # Fidelity National Information Services (SIC 7374)
    1123360: "Financials",              # Global Payments (SIC 7374)
    1018724: "Consumer Discretionary",  # Amazon (SIC 5961 retail - matches, listed for clarity)
}


def sic_to_sector(sic: int | None, cik: int | None = None, is_bdc_like: bool = False) -> str | None:
    if cik is not None and int(cik) in OVERRIDES:
        return OVERRIDES[int(cik)]
    if sic is None or sic != sic:  # None or NaN
        # Operating filers with no SIC are nearly all business development companies.
        return "Financials" if is_bdc_like else None
    sic = int(sic)
    for lo, hi, sector in _RANGES:
        if lo <= sic <= hi:
            return sector
    return None
