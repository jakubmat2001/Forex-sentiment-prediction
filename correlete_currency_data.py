
#link the economic indicator to the corresponding key, this allows us to find economic indicators if they are named in different way 
#then use the key to extract appropriate indicator and it's pip_move value from the "extracted_news_dataset"
economic_indicator_lookup = {
    'cpi': ['cpi', 'consumer price index', 'inflation rate', 'inflationary'],
    'gdp': ['gdp', 'gross domestic product', 'economy grew'],
    'interest rate': ['rate bets','rate lower', 'rate raising','interest rate', 'ir', 'policy rate', 'raising rate', 'rate raising', 'hiked rate', 'hiked rates', 'monetary policy', 'cash rate', 'change policy', 'rate hike','hike rates', 'rate cut', 'rate climb', 'rate drop', 'bsp', 'central bank rates'],
    'unemployment rate': ['unemployment rate', 'ir', 'policy rate', 'employment change', 'jobless rate', 'job', 'employment', 'jobless', 'employment report' , 'participation rate', 'jobs data', 'jobs'],
    'automatic data processing non farm payroll': ['automatic data processing non farm payroll', 'adp', 'adp non farm payroll'],
    'retail sales': ['retail sales', 'sales'],
    'ism manufacturing': ['ism manufacturing', 'ism'],
    'jolts jobs opening': ['jolts jobs opening', 'jolts', 'job', 'jobs'],
    'non-farm-payroll': ['non-farm-payroll','non farm payrolls' 'non farm payroll', 'payroll'],
    'pce price index': ['pce price index', 'ppi'],
    'produce price index': ['produce price index', 'ppi'],
    'consumer sentiment': ['consumer sentiment', 'consumer confidence', 'consumer confidence index'],
    'pce': ['personal consumption expenditures', 'pce'],
    'ppi': ['ppi']
}

#link the currency to the corresponding key, once one currency was found in article text check it's name and look through this dictionary
#if the key isn't either 'united states dollar' or 'australian dollar', sentence is disqualifed even if it has economic_indicator and a pos/neg label
currency_lookup = {
    'united states dollar': ['us', 'american dollar', 'united states dollar', 'jerome powell', 'fomc', 'us dollar', 'greenback', 'fed', 'feds','federal reserve', 'u s', 'u', 'u dollar', 'united states', 'greenback'],
    'australian dollar': ['aussie', 'australian dollar', 'au', 'rba', 'reserve bank of australia', 'aussie dollar', 'australia', 'australian'],
    'united kingdom': ['uk', 'united kingdom', 'bank of england', 'boe'],
    'india': ['india', 'boi', 'bank of india'],
    'korea': ['south korea', 'bok', 'bank of korea'],
    'new zeland': ['new zeland', 'bonz', 'bank of new zealand'],
    'china': ['chinese', 'china', 'bank of china', 'pboc'],
    'euro': ['ecb', 'european central bank'],
    'germany': ['bbk', 'deustsche bundesbank'],
    'japan': ['boj', 'bank of japan', 'japanese', 'japan'],
}

#removes an economic indicator to prevent re-use
usd_indicators = ['cpi', 'gdp','interest rate','unemployment rate','automatic data processing non farm payroll','retail sales' ,'ism manufacturing' ,'jolts jobs opening' ,'non-farm-payroll','pce price index', 'produce price index', 'consumer sentiment' ]
aud_indicators = ['cpi', 'gdp','interest rate','unemployment rate','retail sales' ,'ism manufacturing' ,'pce price index', 'produce price index', 'consumer sentiment' ]