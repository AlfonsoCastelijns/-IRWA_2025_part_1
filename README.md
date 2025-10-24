# -IRWA_2025_part_1

**Part 1**

This part of the IRWA project involves text cleaning, data normalization, and preprocessing of a fashion product dataset, to prepare it for future search.

**Requirements:** The following python packages are used: json, re, nltk, pandas, numpy, matplotlib, seaborn, collections


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

**Function:** `build_metadata_text(doc)`
Creates a new field called metadata_text by combining several metadata attributes into a single string. This field is useful for general-purpose indexing and fallback matching when the query is vague or not tied to a specific field.

  -Combines brand, category, sub_category, product_details, and seller
  -Helps support flexible search and exploratory queries

**Function:** `normalize_numeric_fields(doc)`

Cleans and converts numeric and rating-related fields into proper data
types: - Converts `selling_price` and `actual_price` to `float` -
Extracts integer values from `discount` (e.g., "69% off" → `69`) -
Converts `average_rating` to `float`


For part 2 we don't define any more functions, we use functions that are already created from different libraries, mainly matplotlib and seaborn and the code is executed just by pressing run
