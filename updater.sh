#!/bin/sh
while [ ! -f /app/data/price_data.csv ]; do
  echo 'Waiting for price_data.csv...'
  sleep 10
done
awk -F',' 'NR==1 {print "timestamp,open,high,low,close"} NR>1 {print $1","$17","$18","$19","$20}' /app/data/price_data.csv > /app/data/raw_bera.csv
echo 'Converted price_data.csv to raw_bera.csv'
while true; do
  python -u /app/update_app.py
  sleep 2h
done
