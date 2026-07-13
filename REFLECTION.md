# Reflection

## Producing Matching Results

Producing identical results across Pure Python, Pandas, and Polars required careful attention to several details.

The main areas that required consistency were:

- Missing-value handling
- Whitespace trimming
- Type detection
- Sample standard deviation using `ddof=1`
- Sorting grouped results before comparison
- Handling single-record groups where standard deviation is undefined

A small difference initially appeared in the number of unique `bylines` values because Polars preserved whitespace that the other approaches removed. Applying whitespace trimming resolved the discrepancy.

## Pure Python

Pure Python made every analytical step explicit. I manually implemented numeric conversion, type inference, median calculation, sample standard deviation, categorical frequency counts, and grouped aggregation.

This approach required the most code and was the slowest to develop, but it provided the clearest understanding of how descriptive statistics and grouping work internally.

## Pandas

Pandas provided the most familiar and concise workflow. Functions such as `describe()`, `value_counts()`, `nunique()`, and `groupby()` made the analysis significantly shorter than the Pure Python implementation.

One challenge was that grouped results with multiple aggregations produced MultiIndex columns, which required additional handling when writing and comparing CSV output.

## Polars

Polars used a more expression-oriented approach. Its strict schema and explicit aggregation expressions made the grouped analysis clear and structured.

Polars produced compact and efficient code, although its syntax required more adjustment than Pandas. The output formatting and some API details also differed from Pandas.

## Performance and Developer Experience

Pure Python was the most verbose and required the greatest amount of manual implementation.

Pandas was the easiest approach for rapid exploratory analysis and would be the first library I would recommend to a junior analyst.

Polars would be valuable for larger datasets and workflows where performance, strict typing, and scalable execution are important.

## AI-Assisted Development

AI-generated starter code was useful for creating templates for all three approaches. However, the generated code still required testing and correction.

Examples of issues that needed manual review included:

- Incorrect function placement
- Differences in Pandas CSV headers
- Polars whitespace handling
- Group sorting
- Single-record standard deviation behavior

This reinforced that AI-generated code should be treated as a starting point rather than assumed to be correct.

## Recommendation

For a beginner, I would recommend learning Pandas first because it is concise, widely used, and well documented.

After learning Pandas, I would recommend studying Pure Python implementations to understand the underlying logic, followed by Polars for modern, performance-oriented DataFrame workflows.