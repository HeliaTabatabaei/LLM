import re


def append_image_urls_to_maintext(
    maintext: str,
    imgs_info: list,
    image_base_url: str,
    image_extension: str = ".png",
) -> str:
    """
    image_base_url نمونه:
    http://10.100.52.7:8000/data/2-IT9411-138-00_AudioSystem/img_folder
    """

    if not maintext or not imgs_info or not image_base_url:
        return maintext

    image_base_url = image_base_url.rstrip("/")

    images_by_rid = {
        image.get("rId"): image
        for image in imgs_info
        if image.get("rId")
    }

    def add_image_link(match: re.Match) -> str:
        r_id = match.group(1)
        image_info = images_by_rid.get(r_id)

        if not image_info:
            return match.group(0)

        caption = image_info.get("caption") or f"تصویر {r_id}"
        image_url = f"{image_base_url}/{r_id}{image_extension}"

        return (
            f"img_description_{r_id}:\n"
            f"![{caption}]({image_url})\n"
            f"[نمایش / دانلود تصویر]({image_url})\n"
        )

    return re.sub(
        r"img_description_(rId\d+)\s*:",
        add_image_link,
        maintext
    )
