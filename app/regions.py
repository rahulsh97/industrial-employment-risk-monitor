# Broad region grouping for the Industry-view filter. Keyed by the country name as
# it appears in the UNIDO data. Unknown names fall back to "Other".
REGION = {
    # Europe
    "Albania": "Europe", "Austria": "Europe", "Belarus": "Europe", "Belgium": "Europe",
    "Bosnia and Herzegovina": "Europe", "Bulgaria": "Europe", "Croatia": "Europe",
    "Cyprus": "Europe", "Czechia": "Europe", "Denmark": "Europe", "Estonia": "Europe",
    "Finland": "Europe", "France": "Europe", "Germany": "Europe", "Greece": "Europe",
    "Hungary": "Europe", "Iceland": "Europe", "Ireland": "Europe", "Italy": "Europe",
    "Latvia": "Europe", "Liechtenstein": "Europe", "Lithuania": "Europe", "Luxembourg": "Europe",
    "Malta": "Europe", "Montenegro": "Europe", "Netherlands (Kingdom of the)": "Europe",
    "Norway": "Europe", "Poland": "Europe", "Portugal": "Europe", "Republic of Moldova": "Europe",
    "Romania": "Europe", "Russian Federation": "Europe", "Serbia": "Europe", "Slovakia": "Europe",
    "Slovenia": "Europe", "Spain": "Europe", "Sweden": "Europe", "Switzerland": "Europe",
    "Ukraine": "Europe", "United Kingdom": "Europe",
    # Asia & Pacific
    "Armenia": "Asia & Pacific", "Australia": "Asia & Pacific", "Azerbaijan": "Asia & Pacific",
    "China, Hong Kong SAR": "Asia & Pacific", "Fiji": "Asia & Pacific", "Georgia": "Asia & Pacific",
    "India": "Asia & Pacific", "Japan": "Asia & Pacific", "Kazakhstan": "Asia & Pacific",
    "Kyrgyzstan": "Asia & Pacific", "Mongolia": "Asia & Pacific", "Myanmar": "Asia & Pacific",
    "New Zealand": "Asia & Pacific", "Pakistan": "Asia & Pacific", "Republic of Korea": "Asia & Pacific",
    "Singapore": "Asia & Pacific", "Uzbekistan": "Asia & Pacific", "Viet Nam": "Asia & Pacific",
    # Middle East
    "Iraq": "Middle East", "Oman": "Middle East", "Saudi Arabia": "Middle East",
    "United Arab Emirates": "Middle East", "Türkiye": "Middle East",
    # Africa
    "Angola": "Africa", "Botswana": "Africa", "Cabo Verde": "Africa", "Cameroon": "Africa",
    "Côte d'Ivoire": "Africa", "Djibouti": "Africa", "Guinea-Bissau": "Africa",
    "Kenya": "Africa", "Lesotho": "Africa", "Mauritius": "Africa", "Morocco": "Africa",
    "Namibia": "Africa", "Rwanda": "Africa", "Senegal": "Africa", "Tunisia": "Africa",
    "Zambia": "Africa",
    # Americas
    "Bahamas": "Americas", "Brazil": "Americas", "Canada": "Americas", "Chile": "Americas",
    "Colombia": "Americas", "Costa Rica": "Americas", "Dominican Republic": "Americas",
    "Ecuador": "Americas", "Mexico": "Americas", "Panama": "Americas", "Paraguay": "Americas",
    "Peru": "Americas", "Puerto Rico": "Americas", "United States of America": "Americas",
    "Uruguay": "Americas",
}


def region_of(country):
    return REGION.get(country, "Other")
