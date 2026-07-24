def calculate_final_scores(
    query: str,
    results: list,

):

    semantic_weight: float = 0.9
    keyword_weight: float = 0.1
    query_words = set(
        query.lower().split()
    )

    scored_results = []


    for item in results:

        keywords = " ".join(
            item.get("keywords", [])
        ).lower()

        heading = item.get(
            "heading",
            ""
        ).lower()

        text = item.get(
            "text",
            ""
        ).lower()


        if not query_words:
            keyword_score = 0

        else:

            keyword_match = sum(
                1
                for word in query_words
                if word in keywords
            )


            heading_match = sum(
                1
                for word in query_words
                if word in heading
            )


            text_match = sum(
                1
                for word in query_words
                if word in text
            )


            keyword_score = (
                (keyword_match / len(query_words)) * 0.5
                +
                (heading_match / len(query_words)) * 0.3
                +
                (text_match / len(query_words)) * 0.2
            )

        semantic_score = item.get(
            "score",
            0
        )


        final_score = (
            semantic_score * semantic_weight
            +
            keyword_score * keyword_weight
        )


        item["keyword_score"] = round(
            keyword_score,
            4
        )


        item["final_score"] = round(
            final_score,
            4
        )


        scored_results.append(item)

    scored_results.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )


    return scored_results