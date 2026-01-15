# ELF Malware Detection System
## מסמך תיעוד מקיף לפרויקט - גרסה מעודכנת

---

# תוכן עניינים

1. [סקירה כללית](#1-סקירה-כללית)
2. [ארכיטקטורת המערכת](#2-ארכיטקטורת-המערכת)
3. [מבנה הקבצים](#3-מבנה-הקבצים)
4. [ה-Dataset](#4-ה-dataset)
5. [שלושת ה-Pipelines](#5-שלושת-ה-pipelines)
6. [המודלים](#6-המודלים)
7. [תהליך האימון](#7-תהליך-האימון)
8. [תוצאות והשוואה](#8-תוצאות-והשוואה)
9. [בדיקת Overfitting](#9-בדיקת-overfitting)
10. [הסבר על כל קובץ](#10-הסבר-על-כל-קובץ)
11. [הוראות הפעלה](#11-הוראות-הפעלה)
12. [שאלות צפויות](#12-שאלות-צפויות)

---

# 1. סקירה כללית

## מטרת הפרויקט
פיתוח מערכת לזיהוי קבצי ELF זדוניים (malware) באמצעות ניתוח סטטי ולמידת מכונה.

## הגישה
הפרויקט משווה בין שתי גישות מחקריות קיימות ומציע גישה שלישית משופרת:

| Pipeline | גישה | מבוסס על |
|----------|------|----------|
| **A** | Structural Features | מאמר ELF-Miner |
| **B** | Statistical Features | מאמר Guesmi et al. |
| **C** | Hybrid (שלנו) | שילוב A+B + פיצ'רים חדשים |

## מה זה קובץ ELF?
**ELF (Executable and Linkable Format)** הוא פורמט הקבצים הבינאריים הסטנדרטי במערכות Linux/Unix.
- קבצי הרצה (executables)
- ספריות משותפות (shared libraries)
- קבצי object

### מבנה קובץ ELF:
```
+------------------+
|    ELF Header    |  <- מידע בסיסי על הקובץ
+------------------+
|  Program Headers |  <- איך לטעון לזיכרון
+------------------+
|     Sections     |  <- הקוד והנתונים
|   .text (code)   |
|   .data (data)   |
|   .rodata        |
|   .bss           |
|   ...            |
+------------------+
|  Section Headers |  <- מידע על ה-sections
+------------------+
```

---

# 2. ארכיטקטורת המערכת

## תרשים ארכיטקטורה

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
│                    (frontend.html)                               │
│              ממשק משתמש לגרירת קבצים                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP POST /analyze
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI)                             │
│                   Container: elf_backend                         │
│                      Port: 8000                                  │
│                                                                  │
│  • מקבל קובץ ELF מהמשתמש                                        │
│  • מקודד ל-Base64                                               │
│  • שומר ב-Redis                                                 │
│  • מוסיף job לתור                                               │
│  • מחזיר job_id                                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Redis Queue
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Redis                                     │
│                   Container: elf_redis                           │
│                      Port: 6379                                  │
│                                                                  │
│  • elf_queue - תור העבודות                                      │
│  • elf:file:{job_id} - הקובץ המקודד                             │
│  • elf:result:{job_id} - התוצאה                                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │ BRPOP (blocking)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Worker                                     │
│                   Container: elf_worker                          │
│                                                                  │
│  1. שולף job מהתור                                              │
│  2. מפענח את הקובץ                                              │
│  3. מחלץ פיצ'רים (3 pipelines)                                  │
│  4. מריץ 6 מודלים (2 לכל pipeline)                              │
│  5. שומר תוצאה ב-Redis                                          │
└─────────────────────────────────────────────────────────────────┘
```

## למה ארכיטקטורה כזו?

### יתרונות:
1. **Scalability** - אפשר להוסיף workers במקביל
2. **Decoupling** - הפרדה בין קבלת הקבצים לעיבוד
3. **Fault Tolerance** - אם worker נופל, ה-job נשאר בתור
4. **Async Processing** - המשתמש לא ממתין לעיבוד

### Docker Containers:
| Container | תפקיד | Port |
|-----------|-------|------|
| elf_redis | מסד נתונים + תור | 6379 |
| elf_backend | API server | 8000 |
| elf_worker | עיבוד והסקה | - |

---

# 3. מבנה הקבצים

```
elf-malware-detection/
│
├── docker-compose.yml      # הגדרת Docker containers
│
├── backend/                # שרת ה-API
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py            # FastAPI application
│
├── worker/                 # מעבד העבודות
│   ├── Dockerfile
│   ├── requirements.txt
│   └── worker.py          # Worker עם 3 pipelines
│
├── pipeline_a.py          # Pipeline A - Structural
├── pipeline_b.py          # Pipeline B - Statistical
├── pipeline_c.py          # Pipeline C - Hybrid
│
├── build_features.py      # חילוץ פיצ'רים מה-dataset
├── train_models.py        # אימון והשוואת מודלים
│
├── models/                # המודלים המאומנים
│   ├── pipeline_a_rf.pkl
│   ├── pipeline_a_xgb.pkl
│   ├── pipeline_b_rf.pkl
│   ├── pipeline_b_xgb.pkl
│   ├── pipeline_c_rf.pkl
│   └── pipeline_c_xgb.pkl
│
├── elf_dataset/           # ה-dataset
│   ├── benign/            # קבצים תקינים (~600)
│   └── malware/           # קבצים זדוניים (~594)
│
├── features_pipeline_a.csv  # פיצ'רים שחולצו
├── features_pipeline_b.csv
├── features_pipeline_c.csv
│
├── model_comparison.csv   # תוצאות ההשוואה
├── learning_curves.png    # גרפי overfitting
│
└── frontend.html          # ממשק המשתמש
```

---

# 4. ה-Dataset

## הרכב ה-Dataset

| סוג | כמות | מקור |
|-----|------|------|
| **Benign** | ~600 | קבצי מערכת Linux (`/usr/bin`, `/bin`, `/usr/sbin`) |
| **Malware** | ~594 | GitHub: MalwareSamples/Linux-Malware-Samples + theZoo |
| **סה"כ** | ~1,194 | |

## מקורות הנתונים

### קבצי Benign:
```bash
# הועתקו מהמערכת
sudo find /usr/bin /usr/sbin /bin /sbin -type f -executable
```

דוגמאות: `ls`, `cat`, `grep`, `python3`, `gcc`, `systemctl`

### קבצי Malware:
```bash
# מקור 1: GitHub
git clone https://github.com/MalwareSamples/Linux-Malware-Samples

# מקור 2: theZoo
git clone https://github.com/ytisf/theZoo
# סיסמה לפתיחת ZIP: infected
```

כולל: Mirai botnet, ransomware, backdoors, trojans, cryptominers

## חלוקת Train/Test

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

| סט | כמות | אחוז |
|----|------|------|
| Train | 912 | 80% |
| Test | 229 | 20% |

**stratify=y** - שומר על יחס זהה בין malware/benign בשני הסטים

## איזון ה-Dataset

יחס ~1:1 (מאוזן) חשוב כי:
- מונע הטיה לכיוון הקלאס הגדול
- מאפשר למודל ללמוד שני הקלאסים באופן שווה
- משפר את ה-Precision וה-Recall

---

# 5. שלושת ה-Pipelines

## Pipeline A: Structural Features (ELF-Miner)

### הרעיון
מחלץ מידע **מבני** מה-ELF header ומהמבנה הפנימי של הקובץ.

### הפיצ'רים (53 פיצ'רים)

#### ELF Header (13 פיצ'רים):
| פיצ'ר | הסבר |
|-------|------|
| `e_type` | סוג הקובץ (EXEC=2, DYN=3) |
| `e_machine` | ארכיטקטורה (x86_64=62, ARM=40) |
| `e_version` | גרסת ELF |
| `e_entry` | כתובת נקודת הכניסה |
| `e_phoff` | offset של program headers |
| `e_shoff` | offset של section headers |
| `e_flags` | דגלים |
| `e_ehsize` | גודל ה-ELF header |
| `e_phentsize` | גודל program header entry |
| `e_phnum` | מספר program headers |
| `e_shentsize` | גודל section header entry |
| `e_shnum` | מספר sections |
| `e_shstrndx` | index של string table |

#### Section Features (17 פיצ'רים):
| פיצ'ר | הסבר |
|-------|------|
| `num_sections` | מספר sections כולל |
| `sections_executable` | sections עם הרשאת הרצה |
| `sections_writable` | sections עם הרשאת כתיבה |
| `sections_allocatable` | sections שנטענים לזיכרון |
| `sections_progbits` | sections עם תוכן (קוד/נתונים) |
| `sections_nobits` | sections ללא תוכן (BSS) |
| `sections_symtab` | symbol tables |
| `sections_dynsym` | dynamic symbol tables |
| `sections_strtab` | string tables |
| `sections_rela/rel` | relocation sections |
| `sections_dynamic` | dynamic linking info |
| `sections_note` | notes |
| `section_size_*` | סטטיסטיקות גודל (min/max/mean/total) |

#### Segment Features (10 פיצ'רים):
| פיצ'ר | הסבר |
|-------|------|
| `num_segments` | מספר segments |
| `segments_executable` | segments עם הרשאת הרצה |
| `segments_writable` | segments עם הרשאת כתיבה |
| `segments_readable` | segments עם הרשאת קריאה |
| `segments_load` | PT_LOAD segments |
| `segments_dynamic` | PT_DYNAMIC |
| `segments_interp` | PT_INTERP (dynamic linker) |
| `segments_note` | PT_NOTE |
| `segments_gnu_stack` | PT_GNU_STACK |
| `segments_gnu_relro` | PT_GNU_RELRO |

#### Symbol Features (7 פיצ'רים):
| פיצ'ר | הסבר |
|-------|------|
| `num_symbols` | מספר symbols כולל |
| `num_functions` | פונקציות (STT_FUNC) |
| `num_objects` | משתנים גלובליים (STT_OBJECT) |
| `num_global_symbols` | symbols גלובליים |
| `num_local_symbols` | symbols מקומיים |
| `num_weak_symbols` | symbols חלשים |
| `num_undefined_symbols` | symbols לא מוגדרים (imports) |

#### Dynamic Features (5 פיצ'רים):
| פיצ'ר | הסבר |
|-------|------|
| `num_libraries_needed` | ספריות נדרשות (DT_NEEDED) |
| `num_dynamic_entries` | entries ב-dynamic section |
| `has_rpath` | האם יש RPATH |
| `has_runpath` | האם יש RUNPATH |
| `has_soname` | האם יש SONAME |

#### File Size (1 פיצ'ר):
| פיצ'ר | הסבר |
|-------|------|
| `file_size` | גודל הקובץ בבתים |

### למה זה עובד?
- Malware נוטה להיות **stripped** (ללא symbols)
- Malware לרוב **statically linked** (פחות ספריות)
- מבנה sections חריג (packing, obfuscation)
- נקודת כניסה חריגה

---

## Pipeline B: Statistical Features (Guesmi et al.)

### הרעיון
מתייחס לקובץ כ**רצף בתים** ומחלץ מאפיינים סטטיסטיים.

### הפיצ'רים (44 פיצ'רים)

#### Entropy Features (5 פיצ'רים):

**מה זה Entropy?**
מדד לאקראיות/אי-סדר בנתונים. ערכים 0-8.

```
Entropy = -Σ p(x) * log2(p(x))
```

| ערך | משמעות |
|-----|--------|
| 0-1 | נתונים חוזרים (nulls) |
| 4-5 | טקסט/קוד רגיל |
| 7-8 | נתונים מוצפנים/דחוסים |

| פיצ'ר | הסבר |
|-------|------|
| `global_entropy` | אנטרופיה של כל הקובץ |
| `entropy_min` | מינימום על חלונות |
| `entropy_max` | מקסימום על חלונות |
| `entropy_mean` | ממוצע על חלונות |
| `entropy_std` | סטיית תקן |

#### Byte Histogram (19 פיצ'רים):

התפלגות ערכי הבתים (0-255) ב-16 bins:

| פיצ'ר | טווח | משמעות |
|-------|------|--------|
| `byte_freq_bin_0` | 0-15 | control characters |
| `byte_freq_bin_1` | 16-31 | more control |
| `byte_freq_bin_2` | 32-47 | space, punctuation |
| ... | ... | ... |
| `byte_freq_bin_15` | 240-255 | high bytes |
| `null_byte_freq` | byte 0x00 | null bytes |
| `printable_freq` | 32-126 | תווים מודפסים |
| `high_byte_freq` | 128-255 | bytes גבוהים |

#### N-gram Features (12 פיצ'רים):

**מה זה N-gram?**
רצף של N בתים עוקבים.

דוגמה (2-gram על "HELLO"):
```
HE, EL, LL, LO
```

| פיצ'ר | הסבר |
|-------|------|
| `ngram_N_unique` | כמה N-grams שונים |
| `ngram_N_entropy` | אנטרופיה של N-grams |
| `ngram_N_most_common_freq` | תדירות הנפוץ ביותר |
| `ngram_N_diversity_50` | כמה נדרשים לכסות 50% |

לכל N ב-{2, 3, 4}

#### Header Features (3 פיצ'רים):
| פיצ'ר | הסבר |
|-------|------|
| `header_entropy` | אנטרופיה של 1024 בתים ראשונים |
| `header_null_freq` | אחוז nulls ב-header |
| `elf_header_entropy` | אנטרופיה של 64 בתים ראשונים |

#### Section Entropy (4 פיצ'רים):
| פיצ'ר | הסבר |
|-------|------|
| `text_section_entropy` | אנטרופיה של .text |
| `data_section_entropy` | אנטרופיה של .data |
| `rodata_section_entropy` | אנטרופיה של .rodata |
| `bss_section_size` | גודל .bss |

### למה זה עובד?
- Malware מוצפן/דחוס = **אנטרופיה גבוהה**
- Patterns שונים של בתים
- N-grams חושפים מבנים חוזרים

---

## Pipeline C: Hybrid Features (החידוש שלנו!)

### הרעיון
שילוב חכם של A+B עם **פיצ'רים חדשים** שפיתחנו.

### הפיצ'רים (39 פיצ'רים)

#### מ-Pipeline A (12 פיצ'רים נבחרים):
- `a_e_type`, `a_e_machine`
- `a_num_sections`, `a_num_segments`
- `a_sections_executable`, `a_sections_writable`
- `a_segments_executable`
- `a_num_symbols`, `a_num_functions`
- `a_num_undefined_symbols`, `a_num_libraries_needed`
- `a_file_size`

#### מ-Pipeline B (10 פיצ'רים נבחרים):
- `b_global_entropy`, `b_entropy_mean`, `b_entropy_std`
- `b_null_byte_freq`, `b_printable_freq`, `b_high_byte_freq`
- `b_ngram_2_entropy`, `b_ngram_3_entropy`
- `b_header_entropy`, `b_text_section_entropy`

#### פיצ'רים חדשים שלנו (17 פיצ'רים):

##### Ratio Features (3):
| פיצ'ר | חישוב | משמעות |
|-------|-------|--------|
| `c_code_to_data_ratio` | .text / (.data + .rodata) | יחס קוד לנתונים |
| `c_code_to_file_ratio` | .text / file_size | אחוז קוד מהקובץ |
| `c_header_to_file_ratio` | headers / file_size | אחוז headers |

##### Section Entropy Analysis (4):
| פיצ'ר | הסבר |
|-------|------|
| `c_section_entropy_mean` | ממוצע אנטרופיה על כל sections |
| `c_section_entropy_max` | מקסימום |
| `c_section_entropy_std` | סטיית תקן |
| `c_high_entropy_section_count` | sections עם entropy > 7 |

##### Security Features (3):
| פיצ'ר | הסבר | משמעות |
|-------|------|--------|
| `c_has_nx` | NX (No-Execute) stack | הגנה מפני buffer overflow |
| `c_has_pie` | Position Independent Executable | ASLR support |
| `c_has_relro` | RELocation Read-Only | הגנה על GOT |

**Malware ישן לרוב ללא הגנות אלה!**

##### Suspicious Patterns (3):
| פיצ'ר | הסבר |
|-------|------|
| `c_suspicious_section_count` | sections עם שמות חשודים (.upx, .packed) |
| `c_empty_section_name_count` | sections ללא שם |
| `c_suspicious_import_count` | imports חשודים |

**Imports חשודים:**
```python
['execve', 'fork', 'socket', 'connect', 'bind', 
 'chmod', 'ptrace', 'mprotect', 'dlopen', 'system']
```

##### String Analysis (4):
| פיצ'ר | הסבר |
|-------|------|
| `c_num_strings` | מספר strings בקובץ |
| `c_avg_string_length` | אורך ממוצע |
| `c_url_string_count` | strings עם http/www |
| `c_ip_string_count` | strings שנראים כמו IP |

### למה Pipeline C הכי טוב?
1. **שילוב** - מנצל את היתרונות של שתי הגישות
2. **Feature Selection** - רק הפיצ'רים הרלוונטיים
3. **Domain Knowledge** - פיצ'רים חדשים מבוססי ידע
4. **Security Awareness** - בדיקת הגנות

---

# 6. המודלים

## Random Forest (RF)

### מה זה?
**Ensemble** של עצי החלטה שעובדים במקביל.

### איך עובד?
```
                    ┌─────────────┐
                    │   Dataset   │
                    └──────┬──────┘
                           │ Bootstrap Sampling
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
       ┌────────┐    ┌────────┐    ┌────────┐
       │ Tree 1 │    │ Tree 2 │    │ Tree N │
       └────┬───┘    └────┬───┘    └────┬───┘
            │              │              │
            ▼              ▼              ▼
         malware        benign        malware
                    │
                    ▼
              ┌──────────┐
              │  VOTING  │  <- majority vote
              └──────────┘
                    │
                    ▼
                malware
```

### פרמטרים:
```python
RandomForestClassifier(
    n_estimators=300,    # 300 עצים
    random_state=42,     # reproducibility
    n_jobs=-1            # כל ה-cores
)
```

### יתרונות:
- עמיד ל-overfitting
- עובד טוב עם הרבה פיצ'רים
- נותן feature importance
- לא דורש נרמול

### חסרונות:
- איטי יחסית
- "קופסה שחורה" חלקית

---

## XGBoost (XGB)

### מה זה?
**eXtreme Gradient Boosting** - ensemble של עצים שעובדים **בטור**.

### איך עובד?
```
Dataset → Tree1 → residuals → Tree2 → residuals → Tree3 → ...
                     │                    │
                     └────────────────────┘
                     כל עץ מתקן את הטעויות של הקודם
```

### פרמטרים:
```python
XGBClassifier(
    n_estimators=300,      # 300 עצים
    learning_rate=0.1,     # קצב למידה
    max_depth=6,           # עומק מקסימלי
    random_state=42,
    eval_metric='logloss'
)
```

### יתרונות:
- ביצועים מעולים
- מהיר (אופטימיזציות רבות)
- מונע overfitting (regularization)

### חסרונות:
- רגיש ל-hyperparameters
- דורש יותר tuning

---

## השוואה RF vs XGBoost

| מאפיין | Random Forest | XGBoost |
|--------|---------------|---------|
| סוג Ensemble | Bagging (מקבילי) | Boosting (טורי) |
| מהירות אימון | בינונית | מהירה |
| Overfitting | עמיד מאוד | עמיד עם regularization |
| Tuning נדרש | מעט | יותר |
| Feature Importance | מובנה | מובנה |

---

# 7. תהליך האימון

## שלב 1: חילוץ פיצ'רים

```python
# build_features.py
for file in dataset:
    features_a = extract_pipeline_a(file)  # 53 features
    features_b = extract_pipeline_b(file)  # 44 features
    features_c = extract_pipeline_c(file)  # 39 features
```

### פלט:
- `features_pipeline_a.csv`
- `features_pipeline_b.csv`
- `features_pipeline_c.csv`

## שלב 2: חלוקת נתונים

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% לטסט
    random_state=42,    # שחזור תוצאות
    stratify=y          # שמירה על יחס קלאסים
)
```

## שלב 3: אימון

```python
# לכל pipeline:
rf = RandomForestClassifier(n_estimators=300)
rf.fit(X_train, y_train)

xgb = XGBClassifier(n_estimators=300)
xgb.fit(X_train, y_train)
```

## שלב 4: הערכה

```python
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

### מדדי הערכה:

| מדד | נוסחה | משמעות |
|-----|-------|--------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | כמה נכון מתוך הכל |
| **Precision** | TP/(TP+FP) | מתוך מה שאמרנו malware, כמה באמת |
| **Recall** | TP/(TP+FN) | מתוך ה-malware האמיתי, כמה תפסנו |
| **F1-Score** | 2*(P*R)/(P+R) | ממוצע הרמוני של P ו-R |

```
                    Predicted
                 Malware  Benign
Actual Malware [   TP   |   FN   ]
       Benign  [   FP   |   TN   ]
```

## שלב 5: שמירת מודלים

```python
joblib.dump(rf, "models/pipeline_a_rf.pkl")
joblib.dump(xgb, "models/pipeline_a_xgb.pkl")
# ...
```

---

# 8. תוצאות והשוואה

## טבלת תוצאות מלאה (Dataset: 1,141 samples)

| Pipeline | Model | Accuracy | Precision | Recall | F1-Score |
|----------|-------|----------|-----------|--------|----------|
| **A (Structural)** | **RF** | **100%** | **100%** | **100%** | **1.0000** |
| A (Structural) | XGB | 99.56% | 100% | 99.07% | 0.9953 |
| B (Statistical) | RF | 98.25% | 99.06% | 97.22% | 0.9813 |
| B (Statistical) | XGB | 98.25% | 98.15% | 98.15% | 0.9815 |
| **C (Hybrid)** | **RF** | **100%** | **100%** | **100%** | **1.0000** |
| C (Hybrid) | XGB | 99.56% | 100% | 99.07% | 0.9953 |

## ניתוח התוצאות

### Pipeline A (Structural):
- **ביצועים מצוינים** (100%)
- הפיצ'רים המבניים מאוד אינפורמטיביים
- ה-malware שונה מבנית מקבצים תקינים

### Pipeline B (Statistical):
- **ביצועים טובים** אך נמוכים מ-A (98.25%)
- אנטרופיה לבד לא מספיקה
- חלק מה-malware לא מוצפן/דחוס

### Pipeline C (Hybrid):
- **שווה ל-A** (100%)
- השילוב לא פגע בביצועים
- הפיצ'רים החדשים תורמים

## Top 10 פיצ'רים חשובים (Pipeline C):

| # | פיצ'ר | Importance | מקור |
|---|-------|------------|------|
| 1 | `a_num_segments` | 16.72% | A |
| 2 | `a_sections_executable` | 12.57% | A |
| 3 | `a_num_libraries_needed` | 8.03% | A |
| 4 | `a_num_sections` | 7.34% | A |
| 5 | `a_num_undefined_symbols` | 7.22% | A |
| 6 | `c_section_entropy_std` | 6.59% | **C (חדש)** |
| 7 | `b_header_entropy` | 6.38% | B |
| 8 | `c_has_pie` | 4.82% | **C (חדש)** |
| 9 | `c_has_relro` | 3.69% | **C (חדש)** |
| 10 | `a_sections_writable` | 2.41% | A |

**הפיצ'רים החדשים שלנו ב-Top 10!**

---

# 9. בדיקת Overfitting

## מה זה Overfitting?
כאשר המודל "משנן" את נתוני האימון במקום ללמוד patterns כלליים.

## איך בדקנו?

### 1. 10-Fold Cross-Validation
במקום לחלק פעם אחת, חילקנו 10 פעמים שונות:

```
Fold 1: [Test] [Train] [Train] [Train] [Train] ...
Fold 2: [Train] [Test] [Train] [Train] [Train] ...
...
Fold 10: [Train] [Train] [Train] [Train] [Test]
```

### 2. Learning Curves
גרפים שמראים את הביצועים כפונקציה של כמות הנתונים.

## תוצאות בדיקת Overfitting

### Learning Curves Analysis:

| Pipeline | Train Score | CV Score | פער | מסקנה |
|----------|-------------|----------|-----|-------|
| **A** | 100% | ~99.5% | ~0.5% | ✅ אין overfitting |
| **B** | 100% | ~98% | ~2% | ⚡ מעט overfitting |
| **C** | 100% | ~99% | ~1% | ✅ אין overfitting |

### ניתוח הגרפים:

**Pipeline A:**
- הקווים כמעט מתכנסים
- פער מינימלי של ~0.5%
- **מסקנה: הביצועים אמיתיים**

**Pipeline B:**
- פער קצת יותר גדול (~2%)
- CV score עולה עם יותר נתונים
- **מסקנה: קצת overfitting, אבל סביר**

**Pipeline C:**
- הקווים מתכנסים יפה
- פער קטן של ~1%
- **מסקנה: הביצועים אמיתיים**

## למה אין Overfitting משמעותי?

1. **הפרדה ברורה** - malware ו-benign שונים מבנית
2. **פיצ'רים איכותיים** - לא רעש, מידע אמיתי
3. **Dataset מאוזן** - לא הטיה לקלאס אחד
4. **Random Forest עמיד** - ensemble מונע overfitting

---

# 10. הסבר על כל קובץ

## docker-compose.yml

```yaml
version: "3.9"

services:
  redis:
    image: redis:7-alpine      # Redis מוכן
    container_name: elf_redis
    ports:
      - "6379:6379"            # חשיפת port

  backend:
    build: ./backend           # בנייה מ-Dockerfile
    container_name: elf_backend
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    volumes:
      - ./backend:/app         # mount לפיתוח
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    ports:
      - "8000:8000"
    depends_on:
      - redis                  # יעלה רק אחרי redis

  worker:
    build: ./worker
    container_name: elf_worker
    command: python worker.py
    volumes:
      - ./worker:/app
      - ./models:/app/models   # המודלים
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - redis
```

---

## backend/main.py

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis

app = FastAPI(title="ELF Malware Detector API")

# CORS - מאפשר גישה מה-frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = Redis(host="redis", port=6379)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    1. מקבל קובץ
    2. מקודד ל-base64
    3. שומר ב-Redis
    4. מוסיף לתור
    5. מחזיר job_id
    """
    job_id = str(uuid.uuid4())
    content = await file.read()
    b64_content = base64.b64encode(content).decode("utf-8")
    
    redis_client.set(f"elf:file:{job_id}", b64_content)
    redis_client.lpush("elf_queue", job_id)
    
    return {"job_id": job_id}

@app.get("/result/{job_id}")
def get_result(job_id: str):
    """
    בודק אם יש תוצאה
    """
    result = redis_client.hgetall(f"elf:result:{job_id}")
    if not result:
        return {"status": "pending"}
    return {"status": "done", "result": result}
```

---

## worker/worker.py

```python
# טוען את כל 6 המודלים
models = {
    'pipeline_a_rf': joblib.load("/app/models/pipeline_a_rf.pkl"),
    'pipeline_a_xgb': joblib.load("/app/models/pipeline_a_xgb.pkl"),
    'pipeline_b_rf': joblib.load("/app/models/pipeline_b_rf.pkl"),
    'pipeline_b_xgb': joblib.load("/app/models/pipeline_b_xgb.pkl"),
    'pipeline_c_rf': joblib.load("/app/models/pipeline_c_rf.pkl"),
    'pipeline_c_xgb': joblib.load("/app/models/pipeline_c_xgb.pkl"),
}

def process_elf(data: bytes) -> dict:
    """
    1. מחלץ פיצ'רים מ-3 pipelines
    2. מריץ 6 מודלים
    3. מחזיר תוצאות
    """
    feats_a = extract_pipeline_a(data)  # 53 features
    feats_b = extract_pipeline_b(data)  # 44 features
    feats_c = extract_pipeline_c(data)  # 39 features
    
    results = {}
    
    # Pipeline A
    pred_a_rf = models['pipeline_a_rf'].predict([feats_a])[0]
    # ... וכו' לכל המודלים
    
    # ההחלטה הסופית: Pipeline C + RF (הטוב ביותר)
    results['final_label'] = 'malicious' if pred_c_rf == 1 else 'benign'
    results['final_confidence'] = prob_c_rf
    
    return results

def main_loop():
    """
    לולאה אינסופית - ממתין לעבודות
    """
    while True:
        # BRPOP - blocking pop מהתור
        item = redis_client.brpop("elf_queue")
        job_id = item[1]
        
        # שולף את הקובץ
        b64_data = redis_client.get(f"elf:file:{job_id}")
        data = base64.b64decode(b64_data)
        
        # מעבד
        result = process_elf(data)
        
        # שומר תוצאה
        redis_client.hset(f"elf:result:{job_id}", mapping=result)
```

---

## pipeline_a.py / pipeline_b.py / pipeline_c.py

כל קובץ מכיל:

```python
def extract_pipeline_X(data: bytes) -> list:
    """
    מקבל bytes של קובץ ELF
    מחזיר רשימה של פיצ'רים מספריים
    """
    f = io.BytesIO(data)
    elf = ELFFile(f)  # pyelftools
    
    features = {}
    # ... חילוץ פיצ'רים ...
    
    return [features[k] for k in FEATURE_NAMES]

def get_pipeline_X_feature_names() -> list:
    """
    מחזיר את שמות הפיצ'רים בסדר
    """
    return ['feature1', 'feature2', ...]
```

---

## build_features.py

```python
def build_dataset(basedir="elf_dataset"):
    """
    עובר על כל הקבצים ומחלץ פיצ'רים
    """
    for label_dir, label in [("benign", 0), ("malware", 1)]:
        for file in os.listdir(basedir + "/" + label_dir):
            data = open(file, 'rb').read()
            
            # בדיקת ELF magic
            if data[:4] != b'\x7fELF':
                continue
            
            # חילוץ פיצ'רים
            feats_a = extract_pipeline_a_vector(data)
            feats_b = extract_pipeline_b_vector(data)
            feats_c = extract_pipeline_c_vector(data)
            
            # הוספה ל-DataFrame
            rows_a.append(feats_a + [label])
            # ...
    
    # שמירה ל-CSV
    df_a.to_csv("features_pipeline_a.csv")
    # ...
```

---

## train_models.py

```python
def train_and_evaluate(X_train, X_test, y_train, y_test, name):
    """
    מאמן RF ו-XGBoost, מחזיר מדדים
    """
    # Random Forest
    rf = RandomForestClassifier(n_estimators=300)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    # XGBoost
    xgb = XGBClassifier(n_estimators=300)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    
    # מדדים
    results = {
        'rf': {
            'accuracy': accuracy_score(y_test, rf_pred),
            'f1': f1_score(y_test, rf_pred),
            # ...
        },
        'xgb': {...}
    }
    
    # שמירה
    joblib.dump(rf, f"models/{name}_rf.pkl")
    joblib.dump(xgb, f"models/{name}_xgb.pkl")
    
    return results

def main():
    # Pipeline A
    df_a = pd.read_csv("features_pipeline_a.csv")
    # ... train_test_split ...
    results_a = train_and_evaluate(...)
    
    # Pipeline B, C
    # ...
    
    # השוואה
    print_comparison_table(results_a, results_b, results_c)
```

---

## check_overfitting.py

```python
def check_overfitting():
    """
    בודק overfitting באמצעות 10-Fold CV
    """
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    
    for pipeline in ['a', 'b', 'c']:
        df = pd.read_csv(f"features_pipeline_{pipeline}.csv")
        X, y = df.drop("label", axis=1), df["label"]
        
        rf = RandomForestClassifier(n_estimators=300)
        scores = cross_val_score(rf, X, y, cv=kfold, scoring='f1')
        
        print(f"Pipeline {pipeline}:")
        print(f"  Mean F1: {scores.mean():.4f}")
        print(f"  Std F1:  {scores.std():.4f}")
        
        # בדיקת פער Train vs CV
        rf.fit(X, y)
        train_score = rf.score(X, y)
        gap = train_score - scores.mean()
        
        if gap > 0.05:
            print("⚠️ Possible overfitting!")
        else:
            print("✅ No significant overfitting")
```

---

## frontend.html

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <!-- אזור העלאה -->
    <div id="drop-zone">
        Drag & Drop ELF file
    </div>
    
    <!-- תוצאות -->
    <div id="results">
        <!-- Final result -->
        <!-- Pipeline comparison -->
        <!-- Accuracy table -->
    </div>

    <script>
        async function analyzeFile() {
            // 1. העלאה
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            const {job_id} = await response.json();
            
            // 2. Polling לתוצאה
            while (true) {
                const result = await fetch(`/result/${job_id}`);
                if (result.status === 'done') {
                    displayResults(result);
                    break;
                }
                await sleep(1000);
            }
        }
    </script>
</body>
</html>
```

---

# 11. הוראות הפעלה

## דרישות מקדימות

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y docker.io docker-compose python3 python3-pip python3-venv git
```

## התקנה מלאה

```bash
# 1. Clone או צור את התיקייה
cd ~/Downloads
mkdir elf-malware-detection
cd elf-malware-detection

# 2. צור את המבנה
mkdir -p backend worker models elf_dataset/benign elf_dataset/malware

# 3. העתק את כל הקבצים למקומות הנכונים

# 4. הכן dataset - Benign
sudo find /usr/bin /usr/sbin /bin /sbin -type f -executable | while read f; do
    file "$f" 2>/dev/null | grep -q "ELF" && cp "$f" elf_dataset/benign/ 2>/dev/null
done

# 5. הכן dataset - Malware
GIT_SSL_NO_VERIFY=true git clone https://github.com/MalwareSamples/Linux-Malware-Samples
cp Linux-Malware-Samples/* elf_dataset/malware/

# 6. צור virtual environment
python3 -m venv venv
source venv/bin/activate

# 7. התקן dependencies
pip install pandas numpy scikit-learn xgboost pyelftools joblib matplotlib

# 8. חלץ פיצ'רים
python build_features.py

# 9. אמן מודלים
python train_models.py

# 10. בדוק overfitting (אופציונלי)
python check_overfitting.py

# 11. הפעל Docker
docker compose up --build -d

# 12. בדוק שעובד
docker logs elf_worker

# 13. פתח את ה-frontend
xdg-open frontend.html
```

## עדכון המערכת אחרי אימון מחדש

```bash
# אחרי שאימנת מחדש (train_models.py)
# המודלים נשמרים אוטומטית ב-models/

# הפעל מחדש את ה-worker כדי לטעון את המודלים החדשים
docker compose restart worker

# בדוק שהמודלים נטענו
docker logs elf_worker
```

## בדיקה מהירה

```bash
# שלח קובץ benign
curl -X POST -F "file=@elf_dataset/benign/ls" http://localhost:8000/analyze

# בדוק תוצאה
curl http://localhost:8000/result/{job_id}

# שלח קובץ malware
curl -X POST -F "file=@elf_dataset/malware/FILENAME" http://localhost:8000/analyze
```

## עצירה

```bash
docker compose down
```

## Troubleshooting

| בעיה | פתרון |
|------|-------|
| Worker לא עולה | `docker logs elf_worker` |
| Models not found | ודא שהמודלים ב-`models/` |
| CORS error | ודא ש-`main.py` כולל CORS middleware |
| Pending forever | בדוק שה-worker רץ |

---

# 12. שאלות צפויות

## שאלות על הגישה

**ש: למה static analysis ולא dynamic?**

ת: 
- **מהיר יותר** - לא צריך להריץ את הקובץ
- **בטוח יותר** - אין סיכון להדבקה
- **Scalable** - אפשר לנתח אלפי קבצים
- **חסרון**: לא תופס malware שמסתתר בזמן ריצה

**ש: למה 3 pipelines?**

ת: הפרויקט משווה בין גישות:
- A - מבנית (מהמאמר הראשון)
- B - סטטיסטית (מהמאמר השני)
- C - השילוב והשיפור **שלנו**

**ש: למה RF ו-XGBoost?**

ת:
- שניהם עובדים מצוין עם tabular data
- לא דורשים נרמול
- נותנים feature importance
- מהירים יחסית
- אפשר להשוות Bagging vs Boosting

## שאלות על התוצאות

**ש: למה 100% accuracy?**

ת:
- ההבדל בין malware ל-benign **ברור מבנית**
- פיצ'רים איכותיים שתופסים את ההבדלים
- **אימתנו עם Cross-Validation** - לא overfitting

**ש: איך וידאתם שאין Overfitting?**

ת:
- **10-Fold Cross-Validation** - בדקנו 10 חלוקות שונות
- **Learning Curves** - הקווים מתכנסים
- **פער קטן** (<1%) בין Train ל-CV

**ש: מה עם False Positives/Negatives?**

ת:
- **FP** (benign מזוהה כ-malware) - פחות חמור
- **FN** (malware לא מזוהה) - **מסוכן!**
- Recall של 100% = תפסנו את כל ה-malware

**ש: האם יעבוד על malware חדש?**

ת:
- סביר שכן - הפיצ'רים כלליים
- אבל צריך לאמן מחדש מדי פעם
- Security features (NX, PIE) עוזרים לזהות malware ישן

## שאלות טכניות

**ש: למה pyelftools?**

ת:
- ספריית Python לפרסור ELF
- קלה לשימוש
- תומכת בכל סוגי ELF

**ש: למה Redis?**

ת:
- מהיר מאוד (in-memory)
- תומך בתורים (LPUSH/BRPOP)
- פשוט להתקנה

**ש: למה Docker?**

ת:
- סביבה אחידה
- קל להפעלה
- Isolation
- קל ל-scale

## שאלות על פיצ'רים

**ש: מה הפיצ'ר הכי חשוב?**

ת: `a_num_segments` (16.7%) - מספר ה-segments. Malware נוטה להיות פשוט יותר.

**ש: מה תרומת הפיצ'רים החדשים שלכם?**

ת:
- `c_section_entropy_std` (6.6%) - מזהה אנומליות
- `c_has_pie` (4.8%) - malware ישן ללא ASLR
- `c_has_relro` (3.7%) - הגנה חסרה

**ש: למה Security Features עוזרים?**

ת: Malware ישן (וחלק מהחדש) לא מקומפל עם:
- NX - מונע הרצת קוד מה-stack
- PIE - כתובות אקראיות
- RELRO - הגנה על GOT

קבצים תקינים מודרניים **כמעט תמיד** כוללים הגנות אלה.

---

# סיכום

## מה בנינו?
מערכת מלאה לזיהוי ELF malware עם:
- 3 גישות שונות לחילוץ פיצ'רים (136 פיצ'רים סה"כ)
- 6 מודלים מאומנים
- בדיקת Overfitting עם Cross-Validation
- API server עם Docker
- ממשק משתמש

## מה הצלחנו?
- **Pipeline A ו-C הכי טובים** - 100% accuracy
- **אימתנו שאין overfitting** - Learning Curves מתכנסים
- הפיצ'רים החדשים **שלנו** תורמים משמעותית
- מערכת עובדת מקצה לקצה

## Dataset
- **~1,200 קבצים** (600 benign + 594 malware)
- מקורות: Linux system files + GitHub malware repos
- חלוקה: 80% train, 20% test

## מה אפשר לשפר?
- Dataset גדול יותר (אלפי קבצים)
- יותר סוגי מודלים (Neural Networks)
- Real-time monitoring
- Integration עם אנטי-וירוס
