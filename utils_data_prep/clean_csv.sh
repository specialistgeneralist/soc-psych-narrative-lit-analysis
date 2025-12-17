# helper script to clean CSV files from embedded newlines
iconv -f latin1 -t utf-8 input.csv \
| perl -0777 -pe 's/\n(?!\d+\t\d+\t\d+\t)/ /g' > input_cleaned.csv