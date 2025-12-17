# strip word counts from the normalised CSV
sed -E 's/\([0-9]+ words\)//g' input_cleaned.csv > input_cleaned_nocounts.csv