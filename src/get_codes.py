import json
import time
airport_codes = ['DFW', 'DTW', 'SEA', 'JFK', 'SJC', 'ORD', 'PHX', 'STL', 'LAX',
       'MCO', 'DEN', 'MIA', 'KOA', 'IAH', 'AUS', 'LAS', 'SLC', 'TUS',
       'STT', 'BOS', 'FLL', 'SFO', 'OGG', 'TPA', 'SNA', 'OKC', 'HNL',
       'PHL', 'LGA', 'RDU', 'DCA', 'RIC', 'ATL', 'LBB', 'CLT', 'ELP',
       'SAN', 'BNA', 'JAC', 'SMF', 'EWR', 'IAD', 'LIH', 'SJU', 'ABQ',
       'ORF', 'JAX', 'MSY', 'SAT', 'MCI', 'GUC', 'IND', 'PDX', 'BWI',
       'MSP', 'MKE', 'TUL', 'ONT', 'RSW', 'RNO', 'DSM', 'MFE', 'PSP',
       'OMA', 'EGE', 'PBI', 'SDF', 'PIT', 'FAT', 'DAY', 'STX', 'COS',
       'CMH', 'MTJ', 'HDN', 'BDL', 'MEM', 'CLE', 'HOU', 'BOI', 'OAK',
       'GEG', 'ANC', 'BUF', 'SYR', 'ALB', 'PVD', 'ROC', 'ILM', 'ICT',
       'PWM', 'GSO', 'CHS', 'MDT', 'BHM', 'ADQ', 'BET', 'BRW', 'SCC',
       'FAI', 'JNU', 'KTN', 'YAK', 'CDV', 'SIT', 'PSG', 'WRG', 'OME',
       'OTZ', 'BUR', 'BLI', 'ADK', 'SWF', 'LGB', 'PSE', 'BQN', 'HPN',
       'SAV', 'SRQ', 'BTV', 'ORH', 'DAB', 'CVG', 'BIS', 'AVL', 'GRR',
       'FNT', 'MYR', 'JAN', 'BIL', 'FAR', 'PNS', 'AGS', 'GSP', 'LEX',
       'DAL', 'ATW', 'GPT', 'MLB', 'BZN', 'CAK', 'CHO', 'MSN', 'EYW',
       'TRI', 'LFT', 'ROA', 'ECP', 'VPS', 'XNA', 'EVV', 'AVP', 'MDW',
       'HSV', 'FAY', 'LIT', 'TYS', 'TLH', 'MSO', 'CHA', 'TTN', 'UST',
       'MOB', 'PHF', 'CAE', 'FSD', 'ITO', 'LBE', 'ABE', 'BMI', 'CRW',
       'ACY', 'PPG', 'IAG', 'ACT', 'MLU', 'GRK', 'SHV', 'FSM', 'MAF',
       'SAF', 'JLN', 'LRD', 'BRO', 'TYR', 'GJT', 'YUM', 'DLH', 'GRB',
       'LAN', 'SBA', 'ASE', 'DRO', 'IDA', 'RAP', 'FCA', 'LNK', 'AMA',
       'BFL', 'MLI', 'LSE', 'SBN', 'PSC', 'MOT', 'FLG', 'ISN', 'GFK',
       'GTF', 'FWA', 'MRY', 'MBS', 'PIA', 'SUN', 'TWF', 'SGF', 'CPR',
       'BTR', 'PBG', 'CRP', 'CID', 'SBP', 'RKS', 'CMX', 'MMH', 'PLN',
       'EKO', 'GCC', 'AZO', 'MFR', 'SMX', 'EUG', 'RST', 'TVC', 'SPI',
       'SGU', 'HLN', 'RDM', 'ACV', 'EAU', 'DVL', 'JMS', 'MKG', 'HYS',
       'PAH', 'COD', 'ABR', 'ITH', 'APN', 'ESC', 'BJI', 'MQT', 'CIU',
       'BGM', 'RHI', 'LWS', 'IMT', 'BRD', 'INL', 'PIH', 'GUM', 'HIB',
       'BTM', 'CDC', 'OTH', 'RDD', 'HRL', 'ISP', 'MHT', 'LAR', 'GNV',
       'MEI', 'PIB', 'BPT', 'LAW', 'AEX', 'TXK', 'ROW', 'ERI', 'CLL',
       'HOB', 'LCH', 'SCE', 'CWA', 'OAJ', 'ELM', 'VLD', 'MGM', 'BGR',
       'GTR', 'CSG', 'BQK', 'DHN', 'EWN', 'ABY', 'SPS', 'SJT', 'GGG',
       'ACK', 'MVY', 'HYA', 'BFF', 'WYS', 'GST', 'AKN', 'DLG', 'GCK',
       'MHK', 'ABI', 'GRI', 'EFD', 'PGD', 'SPN', 'ENV']

import urllib.request
for a in airport_codes:
    url = "http://api.wunderground.com/api/3709e0f67dc256ad/geolookup/q/" + str(a) + ".json"
    fp = urllib.request.urlopen(url)
    mybytes = json.loads(fp.read().decode())
    time.sleep(3)
    print (mybytes)
