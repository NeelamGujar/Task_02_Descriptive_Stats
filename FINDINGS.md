# Findings

## Dataset Overview

The Task 02 dataset contains 246,745 rows and 41 columns related to Facebook political advertising during the 2024 U.S. presidential election.

The dataset includes:

- 4,475 unique `page_id` values
- 246,745 unique `ad_id` values
- 31 numeric columns
- 10 non-numeric columns
- 1,009 missing values in `bylines`

## Advertising Scale

The average estimated audience size was approximately 556,463.

The median estimated audience size was 300,000, which was lower than the mean, suggesting that larger-audience ads increased the overall average.

The average estimated number of impressions was approximately 45,602, while the median was 3,499. This large difference indicates a highly right-skewed distribution.

The average estimated spend was approximately $1,061, while the median was only $49. This suggests that most advertisements had relatively low spending levels, while a smaller number of high-spend advertisements increased the average.

## Political Message Characteristics

Approximately:

- 57.28% of ads contained a call-to-action message
- 54.86% contained advocacy messaging
- 38.16% contained issue-focused messaging
- 27.19% contained attack messaging
- 22.27% contained image-focused messaging
- 18.75% contained incivility

## Common Topics

The most common political topics included:

- Economy: 12.21%
- Health: 10.92%
- Social and cultural issues: 10.58%
- Women's issues: 8.09%
- Safety: 3.37%
- Immigration: 3.36%

## Advertiser and Platform Patterns

The most frequent byline was `HARRIS FOR PRESIDENT`, appearing 49,788 times.

The most common publisher platform combination was Facebook and Instagram, representing 214,434 ads, or approximately 86.91% of the dataset.

Most records used USD currency.

## Grouped Analysis

The grouped analysis identified 4,475 unique `page_id` groups.

Some pages contained tens of thousands of advertisements, while many pages contained only one or a few advertisements.

The combined `page_id` and `ad_id` grouping produced 246,745 groups because every `ad_id` was unique. Therefore, each combined group contained one record, and sample standard deviation was undefined for those groups.

## Validation

The Pure Python, Pandas, and Polars implementations produced matching grouped numerical results within the configured floating-point tolerance.