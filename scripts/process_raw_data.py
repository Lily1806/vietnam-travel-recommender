"""
Script xử lý và bóc tách dữ liệu từ bộ vietnam_tourism_v2 JSON.

Đầu vào:
    data/raw/val_vietnam_tourism.json
    data/raw/train_vietnam_tourism.json

Đầu ra:
    data/raw/knowledge_base/*.txt          (tri thức du lịch cho RAG)
    data/raw/destinations.csv              (danh sách địa điểm cho KNN)
    data/raw/reviews.csv                   (tương tác user-item cho KNN)
    data/processed/rag_eval_qa.json        (bộ câu hỏi kiểm tra RAG)

Usage:
    python scripts/process_raw_data.py
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from unicodedata import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
KB_DIR = RAW_DIR / "knowledge_base"

DESTINATION_MAPPING = {
    "Hà Nội: Thủ đô ngàn năm văn hiến": ("Hà Nội", "thành phố cổ", 2, 4.7, "lịch sử,văn hóa,kiến trúc"),
    "Phố cổ Hà Nội: Tham quan bộ": ("Hà Nội", "khu phố cổ", 1, 4.6, "phố cổ,ẩm thực,mua sắm"),
    "Văn Miếu - Quốc Tử Giám": ("Hà Nội", "di tích lịch sử", 1, 4.5, "giáo dục,lịch sử,kiến trúc"),
    "Hồ Hoàn Kiếm và Đền Ngọc Sơn": ("Hà Nội", "danh lam thắng cảnh", 1, 4.8, "hồ,đền,lịch sử"),
    "Lăng Chủ tịch Hồ Chí Minh": ("Hà Nội", "di tích lịch sử", 1, 4.6, "lịch sử,chính trị,tham quan"),
    "Nhà hát Lớn Hà Nội và khu phố Pháp": ("Hà Nội", "kiến trúc Pháp", 2, 4.3, "kiến trúc,văn hóa,Pháp"),
    "Bảo tàng Dân tộc học Việt Nam": ("Hà Nội", "bảo tàng", 1, 4.5, "bảo tàng,văn hóa,dân tộc"),
    "Múa rối nước truyền thống": ("Hà Nội", "nghệ thuật truyền thống", 1, 4.4, "nghệ thuật,truyền thống,biểu diễn"),
    "Vịnh Hạ Long: Kỳ quan thiên nhiên thế giới": ("Quảng Ninh", "vịnh biển", 3, 4.9, "biển,đảo,UNESCO,thiên nhiên"),
    "Du thuyền qua đêm trên vịnh Hạ Long": ("Quảng Ninh", "vịnh biển", 3, 4.8, "du thuyền,biển,đảo,lãng mạn"),
    "Đảo Cát Bà và Vườn quốc gia": ("Hải Phòng", "đảo biển", 2, 4.6, "đảo,trekking,thiên nhiên,lặn biển"),
    "Ninh Bình: Vịnh Hạ Long trên cạn": ("Ninh Bình", "danh lam thắng cảnh", 2, 4.7, "thuyền,hang,chùa,thiên nhiên"),
    "Tràng An và Tam Cốc: Chèo thuyền ngắm cảnh": ("Ninh Bình", "di sản UNESCO", 2, 4.8, "thuyền,hang,UNESCO,phong cảnh"),
    "Hang Múa và điểm ngắm toàn cảnh": ("Ninh Bình", "leo núi", 1, 4.4, "leo núi,phong cảnh,trekking"),
    "Sa Pa: Ruộng bậc thang và văn hóa dân tộc": ("Lào Cai", "núi cao", 2, 4.8, "ruộng bậc thang,dân tộc,núi,trekking"),
    "Chinh phục Fansipan: Nóc nhà Đông Dương": ("Lào Cai", "leo núi", 3, 4.7, "leo núi,cáp treo,kỷ lục,phiêu lưu"),
    "Trekking các bản làng Sa Pa": ("Lào Cai", "trekking", 2, 4.6, "trekking,bản làng,dân tộc,thiên nhiên"),
    "Hành trình mô tô vòng Hà Giang": ("Hà Giang", "mô tô phượt", 2, 4.9, "mô tô,núi,phiêu lưu,phượt"),
    "Cột cờ Lũng Cú và đèo Mã Pì Lèng": ("Hà Giang", "biên giới", 1, 4.7, "biên giới,núi,đèo,phong cảnh"),
    "Thung lũng Mai Châu: Nghỉ dưỡng yên bình": ("Hòa Bình", "thung lũng", 1, 4.5, "thung lũng,homestay,yên tĩnh,thiên nhiên"),
    "Mù Cang Chải mùa lúa chín vàng": ("Yên Bái", "ruộng bậc thang", 1, 4.8, "ruộng bậc thang,nhiếp ảnh,thiên nhiên"),
    "Vườn quốc gia Ba Bể": ("Bắc Kạn", "vườn quốc gia", 1, 4.6, "hồ,rừng,động vật,sinh thái"),
    "Thác Bản Giốc ở biên giới Trung Quốc": ("Cao Bằng", "thác nước", 1, 4.8, "thác,biên giới,thiên nhiên"),
    "Huế: Cố đô Hoàng triều": ("Thừa Thiên Huế", "thành phố lịch sử", 2, 4.6, "lịch sử,cố đô,kiến trúc"),
    "Đại Nội Huế: Kinh thành triều Nguyễn": ("Thừa Thiên Huế", "di tích hoàng cung", 2, 4.7, "hoàng cung,lịch sử,UNESCO"),
    "Lăng tẩm các vua triều Nguyễn": ("Thừa Thiên Huế", "lăng tẩm", 1, 4.5, "lịch sử,lăng,kiến trúc"),
    "Chùa Thiên Mụ bên sông Hương": ("Thừa Thiên Huế", "chùa", 1, 4.7, "chùa,tôn giáo,sông,bình yên"),
    "Du thuyền sông Hương và nhạc cung đình": ("Thừa Thiên Huế", "trải nghiệm văn hóa", 2, 4.6, "thuyền,âm nhạc,văn hóa,sông"),
    "Đà Nẵng: Thành phố năng động bên biển": ("Đà Nẵng", "thành phố biển", 2, 4.7, "biển,thành phố,cầu,hiện đại"),
    "Cầu Vàng và Bà Nà Hills": ("Đà Nẵng", "công viên giải trí", 3, 4.8, "cầu vàng,công viên,núi,sương mù"),
    "Bãi biển Mỹ Khê: Một trong những bãi biển đẹp nhất": ("Đà Nẵng", "bãi biển", 2, 4.7, "biển,bơi lội,cát trắng"),
    "Bán đảo Sơn Trà (Núi Khỉ)": ("Đà Nẵng", "thiên nhiên", 1, 4.6, "rừng,biển,chùa,thiên nhiên"),
    "Ngũ Hành Sơn (Núi Non Nước)": ("Đà Nẵng", "núi đá marble", 1, 4.5, "hang động,tâm linh,nghệ thuật đá"),
    "Phố cổ Hội An": ("Quảng Nam", "phố cổ UNESCO", 2, 4.9, "phố cổ,đèn lồng,UNESCO,lịch sử"),
    "Chùa Cầu Nhật Bản": ("Quảng Nam", "di tích văn hóa", 1, 4.7, "cầu,chùa,Nhật Bản,lịch sử"),
    "May đo quần áo theo yêu cầu tại Hội An": ("Quảng Nam", "dịch vụ mua sắm", 2, 4.6, "may đo,thời trang,thủ công"),
    "Lễ hội đèn lồng Hội An": ("Quảng Nam", "lễ hội", 1, 4.8, "đèn lồng,lễ hội,văn hóa,tâm linh"),
    "Thánh địa Mỹ Sơn: Di tích Chăm Pa": ("Quảng Nam", "di tích Chăm Pa", 1, 4.6, "Chăm Pa,UNESCO,lịch sử,khảo cổ"),
    "Đảo Cù Lao Chàm: Lặn ngắm san hô": ("Quảng Nam", "đảo biển", 2, 4.7, "lặn biển,san hô,đảo,sinh thái biển"),
    "Quy Nhơn: Bờ biển hoang sơ": ("Bình Định", "thành phố biển", 1, 4.6, "biển,hoang sơ,ít khách,yên tĩnh"),
    "Nha Trang: Thành phố biển sôi động": ("Khánh Hòa", "thành phố biển", 2, 4.6, "biển,lặn biển,giải trí,cáp treo"),
    "Tour tham quan các đảo Nha Trang": ("Khánh Hòa", "đảo biển", 2, 4.5, "đảo,thuyền,snorkeling,biển"),
    "Tháp Bà Ponagar": ("Khánh Hòa", "di tích Chăm Pa", 1, 4.4, "Chăm Pa,tâm linh,lịch sử,tháp"),
    "Đà Lạt: Thành phố ngàn hoa": ("Lâm Đồng", "nghỉ dưỡng núi", 2, 4.8, "hoa,núi,mát mẻ,lãng mạn,cà phê"),
    "Các thác nước ở Đà Lạt": ("Lâm Đồng", "thiên nhiên", 1, 4.5, "thác,rừng,picnic,thiên nhiên"),
    "Hồ Xuân Hương và Thung lũng Tình yêu": ("Lâm Đồng", "danh thắng", 1, 4.4, "hồ,công viên,lãng mạn,dạo bộ"),
    "Ga xe lửa cổ Đà Lạt": ("Lâm Đồng", "di tích kiến trúc", 1, 4.3, "ga tàu,kiến trúc Pháp,lịch sử,sống ảo"),
    "Mũi Né: Đồi cát và thể thao nước": ("Bình Thuận", "bãi biển", 2, 4.5, "đồi cát,lướt ván,biển,mạo hiểm"),
    "Suối Tiên Mũi Né": ("Bình Thuận", "danh thắng", 1, 4.3, "suối,đồi cát,thiên nhiên,chụp ảnh"),
    "Vườn quốc gia Phong Nha - Kẻ Bàng": ("Quảng Bình", "hang động", 2, 4.8, "hang động,UNESCO,phiêu lưu,thiên nhiên"),
    "Động Thiên Đường và Động Phong Nha": ("Quảng Bình", "hang động", 2, 4.7, "hang động,thạch nhũ,thuyền,đẹp"),
    "Hang Sơn Đoòng": ("Quảng Bình", "hang động kỷ lục", 3, 5.0, "hang động,kỷ lục thế giới,phiêu lưu,UNESCO"),
    "Vùng phi quân sự (DMZ)": ("Quảng Trị", "di tích chiến tranh", 1, 4.4, "chiến tranh,lịch sử,cầu Hiền Lương,địa đạo"),
    "Thành phố Hồ Chí Minh": ("TP. Hồ Chí Minh", "thành phố đô thị", 2, 4.5, "đô thị,mua sắm,ẩm thực,sôi động"),
    "Dinh Độc Lập": ("TP. Hồ Chí Minh", "di tích lịch sử", 1, 4.6, "lịch sử,chính trị,tham quan"),
    "Bảo tàng Chứng tích Chiến tranh": ("TP. Hồ Chí Minh", "bảo tàng", 1, 4.7, "bảo tàng,chiến tranh,lịch sử,cảm xúc"),
    "Nhà thờ Đức Bà và Bưu điện Trung tâm": ("TP. Hồ Chí Minh", "kiến trúc", 1, 4.5, "nhà thờ,kiến trúc Pháp,chụp ảnh"),
    "Chợ Bến Thành": ("TP. Hồ Chí Minh", "chợ", 1, 4.2, "chợ,mua sắm,ẩm thực,sầm uất"),
    "Địa đạo Củ Chi": ("TP. Hồ Chí Minh", "di tích chiến tranh", 1, 4.6, "địa đạo,chiến tranh,lịch sử,kỳ quan"),
    "Đồng bằng sông Cửu Long": ("Cần Thơ", "đồng bằng sông", 1, 4.6, "sông,vườn cây,miệt vườn,thuyền"),
    "Chợ nổi Cái Răng": ("Cần Thơ", "chợ nổi", 1, 4.7, "chợ nổi,sông,thuyền,đặc sắc"),
    "Khám phá vườn cây trái miền Tây": ("Tiền Giang", "vườn cây trái", 1, 4.4, "vườn,trái cây,sông,sinh thái"),
    "Bến Tre": ("Bến Tre", "miền quê sông nước", 1, 4.4, "dừa,thuyền,làng nghề,yên bình"),
    "Rừng tràm Trà Sư": ("An Giang", "sinh thái", 1, 4.6, "rừng,chim,sinh thái,thiên nhiên,thuyền"),
    "Phú Quốc: Đảo Ngọc": ("Kiên Giang", "đảo du lịch", 3, 4.8, "biển,đảo,resort,cáp treo,lặn biển"),
    "Bãi Sao và Bãi Dài Phú Quốc": ("Kiên Giang", "bãi biển", 2, 4.9, "cát trắng,biển trong,bơi lội,đẹp"),
    "Nhà tù Phú Quốc": ("Kiên Giang", "di tích lịch sử", 1, 4.3, "lịch sử,chiến tranh,tham quan"),
    "Quần đảo An Thới": ("Kiên Giang", "đảo biển", 2, 4.6, "đảo,lặn biển,san hô,snorkeling"),
    "Côn Đảo": ("Bà Rịa - Vũng Tàu", "đảo thiên nhiên", 3, 4.8, "đảo,hoang sơ,rùa biển,lặn biển"),
    "Vũng Tàu": ("Bà Rịa - Vũng Tàu", "thành phố biển", 2, 4.3, "biển,gần Sài Gòn,hải sản,tắm biển"),
    "Tây Ninh: Thánh thất Cao Đài và Núi Bà Đen": ("Tây Ninh", "tâm linh", 1, 4.4, "tôn giáo,Cao Đài,núi,tâm linh"),
}


def slugify(text: str) -> str:
    text = normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:60]


def build_knowledge_base(articles: list, kb_dir: Path) -> int:
    kb_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0
    for article in articles:
        title = article.get("title", "Unknown")
        paragraphs = article.get("paragraphs", [])
        if not paragraphs:
            continue
        lines = [f"# {title}\n"]
        for i, para in enumerate(paragraphs, 1):
            context = para.get("context", "").strip()
            if context:
                lines.append(f"\n## Đoạn {i}\n{context}\n")
        full_text = "\n".join(lines)
        filename = kb_dir / f"{slugify(title)}.txt"
        filename.write_text(full_text, encoding="utf-8")
        files_written += 1
    return files_written


def build_rag_eval_dataset(articles: list, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    qa_records = []
    for article in articles:
        title = article.get("title", "")
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                qa_records.append({
                    "id": qa.get("id", ""),
                    "topic": title,
                    "question": qa.get("question", ""),
                    "answer": qa["answers"][0]["text"] if qa.get("answers") else "",
                    "context": context,
                })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"total": len(qa_records), "data": qa_records}, f, ensure_ascii=False, indent=2)
    return len(qa_records)


def build_destinations_csv(output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["destination_id", "name", "location", "category", "rating", "price_level", "tags"]
    rows = []
    for idx, (title, info) in enumerate(DESTINATION_MAPPING.items(), start=1):
        location, category, price_level, rating, tags = info
        rows.append({
            "destination_id": f"VN{idx:03d}",
            "name": title.split(":")[0].strip(),
            "location": location,
            "category": category,
            "rating": rating,
            "price_level": price_level,
            "tags": tags,
        })
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def build_reviews_csv(destinations_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(destinations_path, "r", encoding="utf-8") as f:
        destinations = list(csv.DictReader(f))

    NUM_USERS = 200
    random.seed(42)
    regions = ["Hà Nội", "Quảng Ninh", "Lào Cai", "Hà Giang", "Ninh Bình",
               "Thừa Thiên Huế", "Đà Nẵng", "Quảng Nam", "Khánh Hòa",
               "Lâm Đồng", "TP. Hồ Chí Minh", "Kiên Giang", "Cần Thơ"]

    user_profiles = []
    for i in range(1, NUM_USERS + 1):
        user_profiles.append({
            "user_id": f"U{i:04d}",
            "preferred_region": random.choice(regions),
            "preferred_types": random.sample(
                ["biển", "núi", "lịch sử", "ẩm thực", "sinh thái", "phiêu lưu", "nghỉ dưỡng"],
                k=random.randint(1, 3)
            ),
            "budget_pref": random.randint(1, 3),
        })

    review_rows = []
    for profile in user_profiles:
        user_id = profile["user_id"]
        n_reviews = random.randint(8, 18)
        preferred, others = [], []
        for dest in destinations:
            if (profile["preferred_region"] in dest["location"] or
                    any(t in dest["tags"] for t in profile["preferred_types"])):
                preferred.append(dest)
            else:
                others.append(dest)

        n_pref = max(1, int(n_reviews * 0.6))
        chosen = (random.sample(preferred, min(n_pref, len(preferred))) +
                  random.sample(others, min(n_reviews - n_pref, len(others))))
        random.shuffle(chosen)

        for dest in chosen:
            base = float(dest["rating"])
            loc_ok = profile["preferred_region"] in dest["location"]
            bud_ok = abs(int(dest["price_level"]) - profile["budget_pref"]) <= 1
            if loc_ok and bud_ok:
                r = base + random.uniform(-0.3, 0.3)
            elif loc_ok or bud_ok:
                r = base + random.uniform(-0.7, 0.2)
            else:
                r = base + random.uniform(-1.2, 0.1)
            review_rows.append({
                "user_id": user_id,
                "destination_id": dest["destination_id"],
                "rating": round(max(1.0, min(5.0, r)), 1),
                "review_text": "",
            })

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "destination_id", "rating", "review_text"])
        writer.writeheader()
        writer.writerows(review_rows)
    return len(review_rows)


def main():
    parser = argparse.ArgumentParser(description="Xử lý dữ liệu Vietnam Tourism JSON")
    parser.add_argument("--val-file", default="data/raw/val_vietnam_tourism.json")
    parser.add_argument("--train-file", default="data/raw/train_vietnam_tourism.json")
    args = parser.parse_args()

    all_articles = []
    for fpath in [args.val_file, args.train_file]:
        p = Path(fpath)
        if p.exists() and p.stat().st_size > 0:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            arts = data.get("data", [])
            all_articles.extend(arts)
            logger.info("Đã đọc %d bài viết từ %s", len(arts), fpath)
        else:
            logger.warning("Bỏ qua (rỗng hoặc không tồn tại): %s", fpath)

    if not all_articles:
        logger.error("Không đọc được dữ liệu!")
        sys.exit(1)

    seen, unique = set(), []
    for a in all_articles:
        if a["title"] not in seen:
            seen.add(a["title"])
            unique.append(a)
    logger.info("Tổng bài viết duy nhất: %d", len(unique))

    logger.info("[1/4] Xây dựng Knowledge Base...")
    n_files = build_knowledge_base(unique, KB_DIR)
    logger.info("  OK: %d file -> %s", n_files, KB_DIR)

    logger.info("[2/4] Tạo bộ câu hỏi đánh giá RAG...")
    eval_path = PROCESSED_DIR / "rag_eval_qa.json"
    n_qa = build_rag_eval_dataset(unique, eval_path)
    logger.info("  OK: %d cap Q&A -> %s", n_qa, eval_path)

    logger.info("[3/4] Tạo destinations.csv cho KNN...")
    dest_path = RAW_DIR / "destinations.csv"
    n_dest = build_destinations_csv(dest_path)
    logger.info("  OK: %d dia diem -> %s", n_dest, dest_path)

    logger.info("[4/4] Tạo reviews.csv (user-item matrix)...")
    reviews_path = RAW_DIR / "reviews.csv"
    n_reviews = build_reviews_csv(dest_path, reviews_path)
    logger.info("  OK: %d luot danh gia -> %s", n_reviews, reviews_path)

    print("\n" + "=" * 60)
    print("XU LY DU LIEU HOAN TAT!")
    print("=" * 60)
    print(f"  Knowledge Base : {n_files} file .txt  ->  {KB_DIR}/")
    print(f"  RAG Eval Q&A   : {n_qa:,} cap         ->  {eval_path}")
    print(f"  Destinations   : {n_dest} dia diem    ->  {dest_path}")
    print(f"  Reviews        : {n_reviews:,} danh gia  ->  {reviews_path}")
    print("=" * 60)
    print("\nBuoc tiep theo: python scripts/build_index.py")


if __name__ == "__main__":
    main()
