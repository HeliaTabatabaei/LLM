from score.caculate_score_final import calculate_final_scores


def test_calculate_keyword_scores():

    query = "خرابی دوربین"


    results = [
        {
            "id": "13",
            "score": 0.558,
            "text": "تصویر ثبت شده در شب سیاه و ناواضح است و نیاز به بررسی نور و لنز دارد.",
            "heading": "عیب‌یابی دوربین - تصویر شب سیاه و ناواضح است",
            "keywords": [
                "تصویر شب",
                "سیاه",
                "ناواضح",
                "دوربین",
                "نور"
            ],
            "date": "1393-03-01"
        },

        {
            "id": "8",
            "score": 0.568,
            "text": "اگر عکس ذخیره نمی‌شود فضای خالی درایو D و نرم افزار دوربین بررسی شود.",
            "heading": "عیب‌یابی دوربین - عکس ذخیره نمی‌شود",
            "keywords": [
                "عکس ذخیره نمی‌شود",
                "درایو D",
                "فضای دیسک",
                "دوربین"
            ],
            "date": "1393-03-01"
        },

        {
            "id": "1",
            "score": 0.596,
            "text": "جمع‌آوری اطلاعات، شناسایی خرابی و رفع عیب دوربین‌های بانک پاسارگاد.",
            "heading": "هدف پروژه",
            "keywords": [
                "هدف پروژه",
                "جمع‌آوری اطلاعات"
            ],
            "date": "1393-03-01"
        }
    ]


    scored = calculate_final_scores(
        query,
        results
    )

    print(scored)
    # for item in scored:
    #     print(
    #         item["id"],
    #         item["heading"],
    #         item["keyword_score"]
    #     )



if __name__ == "__main__":
    test_calculate_keyword_scores()