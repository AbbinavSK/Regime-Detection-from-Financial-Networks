from io import StringIO

import pandas as pd
import requests

from regime_networks import config

FTSE_SECTOR_MAP = {
    "Aerospace & Defence": "IND",
    "Aerospace & defence": "IND",
    "Alternative Energy": "ENE",
    "Automobiles & Parts": "CD",
    "Banking": "FIN",
    "Banking Services": "FIN",
    "Banks": "FIN",
    "Basic Resources": "MAT",
    "Beverages": "CST",
    "Chemicals": "MAT",
    "Collective Investments": "FIN",
    "Collective investments": "FIN",
    "Construction": "IND",
    "Construction & Materials": "IND",
    "Consumer Digital Services": "CD",
    "Consumer Staples": "CST",
    "Containers & packaging": "MAT",
    "Electrical utilities & independent power producers": "UTL",
    "Electricity": "UTL",
    "Electronic & Electrical Equipment": "IND",
    "Electronic equipment & parts": "IND",
    "Energy": "ENE",
    "Engineering and Construction": "IND",
    "Equity Investments": "FIN",
    "Financial Services": "FIN",
    "Financial services": "FIN",
    "Food & Drug Retailers": "CST",
    "Food & drug retailers": "CST",
    "Food & drug retailing": "CST",
    "Food & tobacco": "CST",
    "Food Producers": "CST",
    "Food and drink": "CST",
    "Food, Beverage and Tobacco": "CST",
    "Gas, Water & Multiutilities": "UTL",
    "General Financial": "FIN",
    "General Industrials": "IND",
    "General industrials": "IND",
    "General retailers": "CD",
    "Health Care": "HC",
    "Health Care Equipment & Services": "HC",
    "Health care equipment & supplies": "HC",
    "Hedge Funds": "FIN",
    "Home Construction": "IND",
    "Homebuilding & construction supplies": "IND",
    "Household goods & home construction": "IND",
    "Industrial Engineering": "IND",
    "Industrial engineering": "IND",
    "Industrial Goods and Services": "IND",
    "Industrial goods and services": "IND",
    "Industrial Metals and Mining": "MAT",
    "Industrial Support Services": "IND",
    "Industrial Support services": "IND",
    "Industrial Transportation": "IND",
    "Industrials": "IND",
    "Insurance": "FIN",
    "Investment Banking and Brokerage Services": "FIN",
    "Investment Trusts": "FIN",
    "Investment trusts": "FIN",
    "Leisure Goods": "CD",
    "Life Insurance": "FIN",
    "Life insurance": "FIN",
    "Media": "CD",
    "Mining": "MAT",
    "Mobile telecommunications": "TEL",
    "Multiline utilities": "UTL",
    "Non-life Insurance": "FIN",
    "Nonlife Insurance": "FIN",
    "Oil & Gas Producers": "ENE",
    "Oil & gas producers": "ENE",
    "Oil, Gas and Coal": "ENE",
    "Personal Goods": "CD",
    "Personal goods": "CD",
    "Personal Products": "CST",
    "Pharmaceuticals & Biotechnology": "HC",
    "Pharmaceuticals & biotechnology": "HC",
    "Pharmaceuticals and Biotechnology": "HC",
    "Precious Metals and Mining": "MAT",
    "Real Estate": "RE",
    "Real estate": "RE",
    "Real Estate Investment & Services": "RE",
    "Real Estate Investment Trusts": "RE",
    "Real estate investment trusts": "RE",
    "Retail": "CD",
    "Retail hospitality": "CD",
    "Retailers": "CD",
    "Software & Computer Services": "TECH",
    "Software & computer services": "TECH",
    "Software and Computer Services": "TECH",
    "Support Services": "IND",
    "Support services": "IND",
    "Technology": "TECH",
    "Telecomms": "TEL",
    "Telecommunications": "TEL",
    "Telecommunications services": "TEL",
    "Tobacco": "CST",
    "Travel & Leisure": "CD",
    "Travel & leisure": "CD",
    "Travel and Leisure": "CD",
    "Travel and leisure": "CD",
}

INDEX_SOURCES = {
    "SP500": {
        "name": "S&P 500",
        "wiki_url": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "table_index": 0,
        "symbol_col": "Symbol",
        "sector_col": "GICS Sector",
        "sector_map": {
            "Information Technology": "IT",
            "Health Care": "HC",
            "Financials": "FIN",
            "Consumer Discretionary": "CD",
            "Communication Services": "CS",
            "Industrials": "IND",
            "Consumer Staples": "CST",
            "Energy": "ENE",
            "Utilities": "UTL",
            "Real Estate": "RE",
            "Materials": "MAT",
        },
    },
    "FTSE250": {
        "name": "FTSE 250",
        "wiki_url": "https://en.wikipedia.org/wiki/FTSE_250_Index",
        "table_match": "ftse industry classification benchmark sector",
        "symbol_col": "Ticker",
        "suffix": ".L",
        "sector_map": FTSE_SECTOR_MAP,
    },
    "FTSE100": {
        "name": "FTSE 100",
        "wiki_url": "https://en.wikipedia.org/wiki/FTSE_100_Index",
        "table_match": "ftse industry classification benchmark sector",
        "symbol_col": "Ticker",
        "suffix": ".L",
        "sector_map": FTSE_SECTOR_MAP,
    },
    "Nikkei225": {
        "name": "Nikkei 225",
        "source_url": "https://indexes.nikkei.co.jp/en/nkave/index/component",
        "snapshot": "2026-07-21",
        "suffix": ".T",
        "components": {
            "PHARMA": ["4151", "4502", "4503", "4506", "4507", "4519", "4523", "4568", "4578"],
            "ELECMC": ["285A", "4062", "6479", "6501", "6503", "6504", "6506", "6526", "6645", "6701", "6702",
                "6723", "6724", "6752", "6753", "6758", "6762", "6770", "6841", "6857", "6861", "6902", "6920",
                "6954", "6963", "6971", "6976", "6981", "7735", "7751", "7752", "8035"],
            "AUTO": ["543A", "7201", "7202", "7203", "7211", "7261", "7267", "7269", "7270", "7272"],
            "PRECIS": ["4543", "4902", "6146", "7731", "7733", "7741"],
            "COMMS": ["9432", "9433", "9434", "9984"],
            "BANK": ["5831", "7186", "8304", "8306", "8308", "8309", "8316", "8331", "8354", "8411"],
            "OTHFIN": ["8253", "8591", "8697"],
            "SECUR": ["8601", "8604"],
            "INSUR": ["8630", "8725", "8750", "8766", "8795"],
            "FISH": ["1332"],
            "FOODS": ["2002", "2269", "2282", "2501", "2502", "2503", "2801", "2802", "2871", "2914"],
            "RETAIL": ["3086", "3092", "3099", "3382", "7453", "7532", "8233", "8252", "8267", "9843", "9983"],
            "SERVIC": ["2413", "2432", "3659", "3697", "4307", "4324", "4385", "4661", "4689", "4704", "4751",
                "4755", "6098", "6178", "6532", "7974", "9602", "9735", "9766"],
            "MINING": ["1605"],
            "TEXTIL": ["3401", "3402"],
            "PULP": ["3861"],
            "CHEMIC": ["3405", "3407", "4004", "4005", "4021", "4042", "4043", "4061", "4063", "4183", "4188",
                "4208", "4452", "4901", "4911", "6988"],
            "PETROL": ["5019", "5020"],
            "RUBBER": ["5101", "5108"],
            "GLASS": ["5201", "5214", "5233", "5301", "5332", "5333"],
            "STEEL": ["5401", "5406", "5411"],
            "NONFER": ["3436", "5706", "5711", "5713", "5714", "5801", "5802", "5803"],
            "TRADCO": ["2768", "8001", "8002", "8015", "8031", "8053", "8058"],
            "CONSTR": ["1721", "1801", "1802", "1803", "1808", "1812", "1925", "1928", "1963"],
            "MACHIN": ["5631", "6103", "6113", "6273", "6301", "6302", "6305", "6326", "6361", "6367", "6471",
                "6472", "6473", "7004", "7011", "7013"],
            "SHIPBD": ["7012"],
            "OTHMFG": ["7832", "7911", "7912", "7951"],
            "REALES": ["3289", "8801", "8802", "8804", "8830"],
            "RAILBS": ["9001", "9005", "9007", "9008", "9009", "9020", "9021", "9022"],
            "LANDTR": ["9064", "9147"],
            "MARINE": ["9101", "9104", "9107"],
            "AIRTR": ["9201", "9202"],
            "ELECPW": ["9501", "9502", "9503"],
            "GAS": ["9531", "9532"],
        },
    },
    "CSI300": {
        "name": "CSI 300",
        "wiki_url": "https://en.wikipedia.org/wiki/CSI_300_Index",
        "table_match": "segment",
        "symbol_col": "Ticker",
        "suffix_col": "Exchange",
        "suffix_map": {"Shanghai": ".SS", "Shenzhen": ".SZ"},
        "sector_map": {
            "Information Technology": "IT",
            "Industrials": "IND",
            "Financials": "FIN",
            "Materials": "MAT",
            "Health Care": "HC",
            "Consumer Staples": "CST",
            "Consumer Discretionary": "CD",
            "Utilities": "UTL",
            "Energy": "ENE",
            "Communication": "CS",
            "Real Estate": "RE",
        },
    },
}


# Ticker -> '{SECTOR}_{code}' column label for every constituent of one index, scraped from Wikipedia or read from the hard-coded list.
def fetch_constituents(index_code):
    cfg = INDEX_SOURCES[index_code]

    if "wiki_url" in cfg:
        response = requests.get(cfg["wiki_url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=config.DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))

        if "table_match" in cfg:
            df, sector_col = None, None

            for table in tables:
                columns = [str(column) for column in table.columns]
                match = next(
                    (column for column in columns if column.lower().startswith(cfg["table_match"])),
                    None,
                )

                if cfg["symbol_col"] in columns and match:
                    df, sector_col = table.dropna(subset=[cfg["symbol_col"]]), match
                    break

            if df is None:
                raise ValueError(f"Could not find the constituent table for {cfg['name']}.")
        else:
            df = tables[cfg["table_index"]]
            sector_col = cfg["sector_col"]

        ticker_map = {}

        for _, row in df.iterrows():
            symbol = str(row[cfg["symbol_col"]])

            if "suffix_col" in cfg:
                _, separator, code = symbol.partition(":")

                if not separator:
                    raise ValueError(f"Invalid source symbol: {symbol!r}")

                epic = code.strip().replace(".", "-")
                suffix = cfg["suffix_map"][row[cfg["suffix_col"]]]
            else:
                epic = symbol.replace(".", "-")
                suffix = cfg.get("suffix", "")

            sector_name = row[sector_col]
            sector = cfg["sector_map"].get(sector_name)

            if sector is None:
                sector = "".join(character for character in str(sector_name).upper() if character.isalnum())
                print(f"WARNING: unmapped {cfg['name']} sector {sector_name!r} labelled {sector}; add it to the sector map")

            ticker_map[epic + suffix] = f"{sector}_{epic}"

        print(f"Found {len(ticker_map)} {cfg['name']} constituents")
        return ticker_map

    ticker_map = {f"{code}{cfg['suffix']}": f"{sector}_{code}" for sector, codes in cfg["components"].items() for code in codes}
    print(f"Found {len(ticker_map)} {cfg['name']} constituents")
    return ticker_map
