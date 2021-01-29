# Kelimebot

*AUTHOR*: Emre Okcular

Date : January 21st 2021

![kelimebot](/resources/kelimebot_profile.png)

In this project a Twitter bot is created. The bot tweets word combinations from Turkish dictionary randomly. Even most of the combinations are meaningless sometimes it can tweet meaningful and funny word couples.

<https://twitter.com/kelimebot>

All the words from the turkish TDK dictionary is collected with their types. Total number of words is 75547.

Collected words were paired with their types such as nouns, adjectives, proverbs and verbs.

```
ab,isim
aba,isim
abacı,isim
abacılık,isim
abadi,isim
aba güreşi,isim
abajur,isim
abajurcu,isim
abajurculuk,isim
abajurlu,sıfat
abajursuz,sıfat
abaküs,isim
abalı,sıfat
Abana,isim
abanabilme,isim
abanabilmek,fiil
abandırabilme,isim
...
züppe,sıfat
züppece,zarf
züppeleşme,isim
züppeleşmek,fiil
züppeleştirme,isim
züppeleştirmek,fiil
züppelik,isim
zürafa,isim
zürafagiller,isim
zürefa,isim
zürra,isim
zürriyet,isim
zürriyetli,sıfat
zürriyetsiz,sıfat
zürriyetsizlik,isim
züyuf,isim
züyuf akçe,isim
```

In the below table, you can see which turkish word pairs and conjunctions are used for random combinations.

|  Bağlaçlar | isim + sıfat + bağlaç + isim + sıfat  | zarf + fiil + bağlaç + zarf +fiil  |  isim + sıfat + bağlaç + zarf + fiil    |
|---|:---:|:---:|:---:|
|  ve | *  | *  |   |
|  ile | *  | *  | *  |
|  veya | *  |   |   |
|  yoksa |   |   | *  |
|  ya da | *  | *   |   |
|  hatta |   | *  |   |
|  gibi |  * |   |  * |
|  bir başka deyişle | *  | *  |   |
|  belki de |   | *  |   |
|  veyahut | *  | *  |   |

Here is an example of the timeline.

![kelimebot](/resources/keilmebot_timeline.png)

You can check the [github repository](https://github.com/emreokcular/kelimebot) for the source code.

For future work, more meaningful tweets will be generated with NLP techniques.