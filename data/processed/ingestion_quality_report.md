# Ingestion Quality Report

## Global
- Registry documents: 4
- Chunks: 527
- Extraction methods: `{'ocr_tesseract': 458, 'pymupdf': 68, 'ocr_tesseract,pymupdf': 1}`
- Levels: `{'preamble': 3, 'article': 236, 'clause': 219, 'point': 57, 'document': 6, 'table': 6}`
- Missing metadata: `{'missing_doc_id': 0, 'missing_page': 0, 'missing_source_path': 0, 'missing_source_url': 0, 'missing_article_label': 0, 'missing_parent_id': 0, 'missing_table_id': 0}`
- Chunk chars: min=201, avg=843.4, max=2199

## By Document
### 45_2019_qh14
- Source: `data/raw/45_2019_QH14.pdf`
- Source URL: `https://luatvietnam.vn/lao-dong/bo-luat-lao-dong-2019-so-45-2019-qh14-179015-d1.html`
- Chunks/pages/chars: 273 chunks, pages 1-83, 244872 chars
- Methods: `{'ocr_tesseract': 273}`
- Suspicious per 10k chars: `{'replacement_char': 0.0, 'mojibake_marker': 9.76, 'ocr_digit_in_word': 28.75, 'broken_legal_heading': 0.04}`
- Short/long chunks: 0 short, 0 long
- Structure chunks: article=185, clause=81, point=6, table=0
- Missing article labels / parent ids: 0 / 0
- Flagged samples:
  - `45_2019_qh14_c00272` p.83-83 ocr_tesseract score=45: Văn bản: 45 2019 QH14 Số hiệu: 45/2019/QH14 Chương XV. H Mục 5. ĐÌNH CÔNG Điều 220. Hiệu lực thi hành Cấp chunk: article Điều 220. Hiệu lực thi hành 1, Bộ luật này có hiệu lực thi hành từ ngày 01 tháng 01 năm 2021. Bộ luật Lao động số 10/2012/QH13 hết hiệu lực thi hành kể từ ngày
  - `45_2019_qh14_c00194` p.60-60 ocr_tesseract score=41: Văn bản: 45 2019 QH14 Số hiệu: 45/2019/QH14 Chương X. H Mục 6. MỘT SỐ LAO ĐỘNG KHÁC Điều 168. Tham gia bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp Khoản 3 Cấp chunk: clause Điều 168. Tham gia bảo hiểm xã hội, bảo hiểm y tế, bảo hiểm thất nghiệp 3. Đối với người lao động 
  - `45_2019_qh14_c00204` p.63-64 ocr_tesseract score=40: Văn bản: 45 2019 QH14 Số hiệu: 45/2019/QH14 Chương X. HI Mục 6. MỘT SỐ LAO ĐỘNG KHÁC Điều 176. Quyền của thành viên ban lãnh fflm của tổ chức đại diện Cấp chunk: article Điều 176. Quyền của thành viên ban lãnh fflm của tổ chức đại diện người lao động tại cơ sở 1, Thành viên ban l

### luat_109_2025_qh15_pdf
- Source: `data/raw/Luat 109_2025_QH15 PDF.pdf`
- Source URL: `https://luatvietnam.vn/thue/luat-thue-thu-nhap-ca-nhan-2025-so-109-2025-qh15-422733-d1.html`
- Chunks/pages/chars: 65 chunks, pages 1-15, 42907 chars
- Methods: `{'pymupdf': 56, 'ocr_tesseract,pymupdf': 1, 'ocr_tesseract': 8}`
- Suspicious per 10k chars: `{'replacement_char': 0.0, 'mojibake_marker': 6.99, 'ocr_digit_in_word': 32.63, 'broken_legal_heading': 0.0}`
- Short/long chunks: 0 short, 0 long
- Structure chunks: article=26, clause=38, point=0, table=0
- Missing article labels / parent ids: 0 / 0
- Flagged samples:
  - `luat_109_2025_qh15_pdf_c00064` p.14-15 ocr_tesseract score=58: Văn bản: Luat 109 2025 QH15 PDF Số hiệu: 109/2025/QH15 Phần thu. nhập tính Chương IV. ĐIỀU KHOẢN THIỊ HÀNH Điều 29. Hiệu lực thi hành Cấp chunk: article Điều 29. Hiệu lực thi hành l. Luật này có hiệu lực thi hành từ ngày 01 tháng 7 năm 2026, trừ trường hợp quy định tại khoản 2 Đi
  - `luat_109_2025_qh15_pdf_c00019` p.4-4 pymupdf score=26: Văn bản: Luat 109 2025 QH15 PDF Số hiệu: 109/2025/QH15 Chương I. NHỮNG QUY ĐỊNH CHUNG Điều 4. Thu nhập được miễn thuế Khoản 6 Cấp chunk: clause Điều 4. Thu nhập được miễn thuế 6. Thu nhập từ lãi trái phiếu chính phủ, lãi trái phiếu chính quyền địa phương, lãi tiền gửi tại tổ chức
  - `luat_109_2025_qh15_pdf_c00000` p.1-1 pymupdf score=21: QUỐC HỘI Luật số: /2025/QH15 CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập - Tự do - Hạnh phúc LUẬT THUẾ THU NHẬP CÁ NHÂN Căn cứ Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam đã được sửa đổi, bổ sung một số điều theo Nghị quyết số 203/2025/QH15; Quốc hội ban hành Luật Thuế thu n

### luat_116_2025_qh15_pdf
- Source: `data/raw/Luat 116_2025_QH15 PDF.pdf`
- Source URL: `https://luatvietnam.vn/thong-tin/luat-an-ninh-mang-2025-so-116-2025-qh15-422396-d1.html#`
- Chunks/pages/chars: 177 chunks, pages 1-37, 142741 chars
- Methods: `{'ocr_tesseract': 177}`
- Suspicious per 10k chars: `{'replacement_char': 0.0, 'mojibake_marker': 13.31, 'ocr_digit_in_word': 32.37, 'broken_legal_heading': 0.0}`
- Short/long chunks: 0 short, 0 long
- Structure chunks: article=25, clause=100, point=51, table=0
- Missing article labels / parent ids: 0 / 0
- Flagged samples:
  - `luat_116_2025_qh15_pdf_c00176` p.36-37 ocr_tesseract score=69: Văn bản: Luat 116 2025 QH15 PDF Số hiệu: 116/2025/QH15 Chương VIII. ĐIỀU KHOẢN THỊ HÀNH Điều 45. Điều khoản chuyên tiếp Cấp chunk: article Điều 45. Điều khoản chuyên tiếp 1. Hệ thống thông tin đã được xác định cấp độ theo quy định của Luật An toàn thông tin mạng số 86/2015/QH13 đ
  - `luat_116_2025_qh15_pdf_c00162` p.34-34 ocr_tesseract score=67: Văn bản: Luat 116 2025 QH15 PDF Số hiệu: 116/2025/QH15 Chương VII. TRÁCH NHIỆM CUA CƠ QUAN, TỎ CHỨC, CÁ NHÂN Điều 42. Trách nhiệm của cơ quan, tổ chức, cá nhân sử dụng không Khoản 3 Cấp chunk: clause Điều 42. Trách nhiệm của cơ quan, tổ chức, cá nhân sử dụng không gian mạng 3. Th
  - `luat_116_2025_qh15_pdf_c00170` p.36-36 ocr_tesseract score=45: Văn bản: Luat 116 2025 QH15 PDF Số hiệu: 116/2025/QH15 Chương VII. TRÁCH NHIỆM CUA CƠ QUAN, TỎ CHỨC, CÁ NHÂN Điều 42. Trách nhiệm của cơ quan, tổ chức, cá nhân sử dụng không Khoản 12 Cấp chunk: clause Điều 42. Trách nhiệm của cơ quan, tổ chức, cá nhân sử dụng không gian mạng 12. 

### phu_luc_ii_kem_nq_bgd_hcm_cu_1
- Source: `data/raw/PHU-LUC-II-KEM-NQ-BGD-HCM-cu-1.pdf`
- Source URL: `https://cdn.thuvienphapluat.vn/uploads/tintuc/2025/11/10/PHU-LUC-II-KEM-NQ-BGD-HCM-cu-1%20(1).pdf`
- Chunks/pages/chars: 12 chunks, pages 1-6, 14480 chars
- Methods: `{'pymupdf': 12}`
- Suspicious per 10k chars: `{'replacement_char': 0.0, 'mojibake_marker': 2.07, 'ocr_digit_in_word': 0.0, 'broken_legal_heading': 0.0}`
- Short/long chunks: 0 short, 0 long
- Structure chunks: article=0, clause=0, point=0, table=6
- Missing article labels / parent ids: 0 / 0
- Flagged samples:
  - `phu_luc_ii_kem_nq_bgd_hcm_cu_1_c00000` p.1-1 pymupdf score=10: TỪ ĐẾN ĐẤT Ở (1) (2) (3) (4) (5) 1 ALEXANDRE DE RHODES TRỌN ĐƯỜNG 430.400 2 CAO BÁ QUÁT TRỌN ĐƯỜNG 294.100 3 CHU MẠNH TRINH TRỌN ĐƯỜNG 326.500 4 CÔNG TRƯỜNG LAM SƠN TRỌN ĐƯỜNG 491.700 5 CÔNG TRƯỜNG MÊ LINH TRỌN ĐƯỜNG 450.800 6 CÔNG XÃ PARIS TRỌN ĐƯỜNG 450.800 7 ĐINH TIÊN HOÀNG LÊ
  - `phu_luc_ii_kem_nq_bgd_hcm_cu_1_c00006` p.1-1 pymupdf score=5: Văn bản: PHU-LUC-II-KEM-NQ-BGD-HCM-cu-1 Số hiệu: Bảng: phu_luc_ii_kem_nq_bgd_hcm_cu_1_p1_t1 Trang: 1 Cấp chunk: table | STT | TÊN ĐƯỜNG | ĐOẠN ĐƯỜNG | | | | --- | --- | --- | --- | --- | | | | TỪ | ĐẾN | ĐẤT Ở | | (1) | (2) | (3) | (4) | (5) | | 1 | ALEXANDRE DE RHODES | TRỌN ĐƯỜ
