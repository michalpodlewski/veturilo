# veturilo
Processing and analysis of data crawled from Warsaw bike sharing system.

The data was gathered in simplest possible way - using wget in a cron job every 10 minutes (+zip what was crawled every 24 hours).
That's why some additional processing is needed, mainly: 
- extracting json from HTML and converting it to dataframes
- gathering daily dataframes into monthly batches

The processing is handled by the script veturilo_extractor.py

Due to hosting limitations I deliberately don't put location of the data here but if you are interested in obtaining the data in either raw or processed form feel free to contact me. 

The module veturilo_helper.py contains a few functions for further data processing i.e.:
- identifying rentals
- aggregating hourly number of available bikes 
- and more to come (creating graph of rides between stations etc.)


