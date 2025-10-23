# -IRWA_2025_part_1

**Part 1**

This part of the IRWA project involves text cleaning, data normalization, and preprocessing of a fashion product dataset, to prepare it for future search.

**Requirements:** The following python packages are used: nltk


In the file IRWA_part1.ipynb, we created multiple functions and code to do the tasks asked:


**Function:** `clean_text(text)`

-   Converts text to lowercase
-   Removes non-ASCII characters and punctuation
-   Collapses multiple spaces
-   Removes stopwords (using NLTK English stopwords)
-   Applies stemming (Porter Stemmer)


**Function:** `ensure_fields(doc)`

Ensures every document has the required set of fields:

``` python
REQUIRED_FIELDS = [
  'pid', 'title', 'description', 'brand', 'category', 'sub_category',
  'product_details', 'seller', 'out_of_stock', 'selling_price',
  'discount', 'actual_price', 'average_rating', 'url'
]
```
If any are missing, they are filled with `None`.


**Function:** `normalize_numeric_fields(doc)`

Cleans and converts numeric and rating-related fields into proper data
types: - Converts `selling_price` and `actual_price` to `float` -
Extracts integer values from `discount` (e.g., "69% off" → `69`) -
Converts `average_rating` to `float`
