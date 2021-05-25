# TGIF Analysis

*Author*: Emre Okcular

Date Created: Mat 21st 2021

<center><img src="/resources/weekend_loading.jpeg" width="50%" and height="50%"></center>

Have you ever wonder which weekend is better than the others ? Indeed we always thankful for the friday but is it true that we wait some weekends more than the others ? Might be a seasonal trend about our weekend expectations ? Any difference between locations ? How did the trend of Positive and negative thoughts change about the weekends ?

To answer all that questions, tweets from the twitter account [@craigweekend](https://twitter.com/craigweekend) collected and analyzed.

<center><img src="/resources/craigweekend_profile.png" width="50%" and height="50%"></center>

This bot account tweets the same video of Daniel Craig announces The Weekend. It tweets all the tweets in Friday afternoon Pacific Time.

<center><img src="/resources/craig.gif" width="50%" and height="50%"></center>

First lets collect all the tweets which are not a reply or retweet with Tweepy API.

Below you can see the dataset that is scraped.

|    |                  id | created             |   favorite |   retweeted | text                    |
|---:|--------------------:|:--------------------|-----------:|------------:|:------------------------|
|  0 | 1395879326399156225 | 2021-05-21 23:09:00 |     214676 |       40972 | https://t.co/2SkLzdr2pH |
|  1 | 1393340094602366976 | 2021-05-14 22:59:00 |     202994 |       37604 | https://t.co/7cuUpCvfMx |
|  2 | 1390809419538128896 | 2021-05-07 23:23:00 |     163499 |       33795 | https://t.co/hHeaj0b7Xq |
|  3 | 1388273710918881280 | 2021-04-30 23:27:00 |     155688 |       31437 | https://t.co/zQSF0EQPLg |
|  4 | 1385729697783312386 | 2021-04-23 22:58:00 |     118392 |       28595 | https://t.co/tfoCb4VxJc |

The basic visualization for retweet and favorite counts looks like below. It is clear that the account became viral in January 2021 and gained followers. There is a dramatic decrease in first weekend of February. Finally, the expectation for the weekends decreased from February to May.

<center><img src="/resources/craig_trend.png" width="50%" and height="50%"></center>