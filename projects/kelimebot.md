# Kelimebot

*AUTHOR*: Emre Okcular

Date : January 21st 2021

![kelimebot](/resources/kelimebot_profile.png)

In this project a Twitter bot is created. The bot tweets word combinations from Turkish dictionary randomly. Even most of the combinations are meaningless sometimes it can tweet meaningful and funny word couples.

<https://twitter.com/kelimebot>

All the words from the turkish TDK dictionary is collected with their types. Total number of words is 75547.

Endpoint : ```https://sozluk.gov.tr/gts```
Example Request : ```https://sozluk.gov.tr/gts?ara=arayış```
Response :
```
[
  {
    "madde_id": "7557",
    "kac": "0",
    "kelime_no": "2831",
    "cesit": "0",
    "anlam_gor": "0",
    "on_taki": null,
    "madde": "arayış",
    "cesit_say": "0",
    "anlam_say": "1",
    "taki": null,
    "cogul_mu": "0",
    "ozel_mi": "0",
    "lisan_kodu": "0",
    "lisan": "",
    "telaffuz": null,
    "birlesikler": null,
    "font": null,
    "madde_duz": "arayis",
    "gosterim_tarihi": null,
    "anlamlarListe": [
      {
        "anlam_id": "5339",
        "madde_id": "7557",
        "anlam_sira": "1",
        "fiil": "0",
        "tipkes": "0",
        "anlam": "Arama işi",
        "gos": "0",
        "orneklerListe": [
          {
            "ornek_id": "16655",
            "anlam_id": "5339",
            "ornek_sira": "1",
            "ornek": "Meydan tatminsizlerin tatmin arayışlarına mı kalırdı?",
            "kac": "1",
            "yazar_id": "7",
            "yazar": [
              {
                "yazar_id": "7",
                "tam_adi": "Tarık Buğra",
                "kisa_adi": "T. Buğra",
                "ekno": "132"
              }
            ]
          }
        ],
        "ozelliklerListe": [
          {
            "ozellik_id": "19",
            "tur": "3",
            "tam_adi": "isim",
            "kisa_adi": "a.",
            "ekno": "30"
          }
        ]
      }
    ]
  }
]
```

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