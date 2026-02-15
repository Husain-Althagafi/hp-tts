arabic_tts_test_sentences = [
    # 1. Basic Clarity (MSA)
    "السلام عليكم، كيف يمكنني مساعدتك اليوم؟",
    "نشكرك على تواصلك معنا.",
    "يرجى الانتظار لحظة من فضلك.",
    "هل يمكن أن توضح طلبك أكثر؟",
    "يسعدني خدمتك.",

    # 2. Numbers
    "رقم الطلب هو ١٢٣٤٥.",
    "المبلغ المستحق هو ٣٥٠ ريالاً.",
    "سيتم تحويل المبلغ خلال ٢٤ ساعة.",
    "رقم الهوية ينتهي بالرقم ٧٨٩٠.",
    "مدة الاشتراك هي ١٢ شهراً.",
    "رقم الطلب هو 12345.",
    "سيتم التواصل خلال 48 ساعة.",

    # 3. Dates and Times
    "الموعد المحدد هو يوم الثلاثاء الساعة الثالثة مساءً.",
    "سيتم الشحن بتاريخ 15 نوفمبر 2025.",
    "وقت العمل من الساعة التاسعة صباحاً حتى الخامسة مساءً.",

    # 4. Mixed Arabic + English
    "سيتم إرسال رسالة SMS خلال دقائق.",
    "يمكنك تحميل التطبيق من App Store أو Google Play.",
    "يرجى إدخال رقم OTP المرسل إلى هاتفك.",
    "تم تحديث حالة الطلب في النظام.",
    "رقم التذكرة هو Ticket 4567.",

    # 5. Long Natural Sentences
    "نفهم تماماً استفسارك، وسنعمل على حل المشكلة بأسرع وقت ممكن لضمان رضاك الكامل عن الخدمة.",
    "نعتذر عن أي إزعاج قد تسبب به هذا الخطأ، ونسعى لتحسين تجربتك معنا باستمرار.",

    # 6. Confirmation / Repetition
    "للتأكيد، هل تقصد أنك ترغب في إلغاء الاشتراك؟",
    "إذن سيتم تحديث العنوان إلى حي النزهة، صحيح؟",
    "هل ترغب في استلام الفاتورة عبر البريد الإلكتروني؟",

    # 7. Short Acknowledgements
    "نعم.",
    "بالطبع.",
    "بالتأكيد.",
    "لحظة من فضلك.",
    "تم.",

    # 8. Dialect (Gulf/Saudi)
    "وش المشكلة اللي تواجهها؟",
    "خلني أتأكد لك من الطلب.",
    "أبشر، راح أشيك على الموضوع.",
    "لحظة بس أتأكد من النظام.",
    "تم حل المشكلة إن شاء الله.",

    # 9. Edge Case Pronunciation
    "شكراً لاختياركم شركتنا.",
    "إجراءات الاسترجاع والاستبدال.",
    "الاستحقاقات والاشتراكات.",
    "خدمات الاتصالات السلكية واللاسلكية.",
    "مسؤول خدمة العملاء.",

    # 10. Stress Test
    "بناءً على البيانات المسجلة لدينا، تم تأكيد الطلب وإرساله إلى قسم الشحن، وسيصلك إشعار فور خروجه من المستودع."
]

english_test_set = [
    # --- Greetings / small talk ---
    {"id": "en_001", "category": "greeting", "text": "Hello, how are you today?"},
    {"id": "en_002", "category": "greeting", "text": "Good morning, can you hear me clearly?"},
    {"id": "en_003", "category": "greeting", "text": "Hi, I just wanted to ask a quick question."},
    {"id": "en_004", "category": "greeting", "text": "Thanks for your help, I really appreciate it."},

    # --- Simple requests / support-style ---
    {"id": "en_010", "category": "request", "text": "I need help with my account, please."},
    {"id": "en_011", "category": "request", "text": "Can you check the status of my order?"},
    {"id": "en_012", "category": "request", "text": "I want to cancel my subscription starting next month."},
    {"id": "en_013", "category": "request", "text": "Please update my phone number on the account."},

    # --- Numbers & money (very important) ---
    {"id": "en_020", "category": "numbers", "text": "My order number is one two three four five six."},
    {"id": "en_021", "category": "numbers", "text": "The total amount is three hundred and fifty dollars."},
    {"id": "en_022", "category": "numbers", "text": "I paid forty nine ninety nine last month."},
    {"id": "en_023", "category": "numbers", "text": "My phone number is zero five five one two three four five six seven."},
    {"id": "en_024", "category": "numbers", "text": "The last four digits of my card are nine eight seven six."},

    # --- Dates & times ---
    {"id": "en_030", "category": "datetime", "text": "I placed the order on the fifteenth of January."},
    {"id": "en_031", "category": "datetime", "text": "Can we schedule a call tomorrow at three thirty p m?"},
    {"id": "en_032", "category": "datetime", "text": "The appointment was originally on March twenty second."},
    {"id": "en_033", "category": "datetime", "text": "I need the delivery before next Monday morning."},

    # --- Addresses / entities / spelling ---
    {"id": "en_040", "category": "identity", "text": "My name is Ahmed Al Thagafi."},
    {"id": "en_041", "category": "identity", "text": "The email address is ahmed dot test at example dot com."},
    {"id": "en_042", "category": "identity", "text": "I live at twenty one Baker Street, apartment number five."},
    {"id": "en_043", "category": "identity", "text": "The city is Jeddah and the postal code is two one five seven seven."},

    # --- Mixed / hesitation / realistic speech ---
    {"id": "en_050", "category": "natural_speech", "text": "Uh, I think there is a mistake on my last bill."},
    {"id": "en_051", "category": "natural_speech", "text": "Sorry, could you please repeat that more slowly?"},
    {"id": "en_052", "category": "natural_speech", "text": "I was disconnected earlier and I'm calling back now."},
    {"id": "en_053", "category": "natural_speech", "text": "I'm not sure which plan I'm currently on."},

    # --- Longer sentences (stress ASR + TTS) ---
    {"id": "en_060", "category": "long", "text": "I received an email saying my account was suspended, but when I log in everything looks normal, so I'm a bit confused."},
    {"id": "en_061", "category": "long", "text": "Before I make any changes, I just want to confirm that there will be no extra charges added to my monthly payment."},
    {"id": "en_062", "category": "long", "text": "If possible, I'd like you to summarize the main differences between my current plan and the premium plan."},
]



arabic_test_set = [

    # --- Greetings / Basic clarity ---
    {"id": "ar_001", "category": "greeting", "text": "السلام عليكم، كيف يمكنني مساعدتك اليوم؟"},
    {"id": "ar_002", "category": "greeting", "text": "أهلاً وسهلاً، هل تسمعني بوضوح؟"},
    {"id": "ar_003", "category": "greeting", "text": "شكراً لتواصلك معنا."},
    {"id": "ar_004", "category": "greeting", "text": "يسعدني خدمتك اليوم."},

    # --- Support-style requests ---
    {"id": "ar_010", "category": "request", "text": "أحتاج إلى مساعدة بخصوص حسابي."},
    {"id": "ar_011", "category": "request", "text": "هل يمكنك التحقق من حالة طلبي؟"},
    {"id": "ar_012", "category": "request", "text": "أرغب في إلغاء الاشتراك بدءاً من الشهر القادم."},
    {"id": "ar_013", "category": "request", "text": "يرجى تحديث رقم الهاتف المسجل في الحساب."},

    # --- Numbers (digit form) ---
    {"id": "ar_020", "category": "numbers", "text": "رقم الطلب هو 12345."},
    {"id": "ar_021", "category": "numbers", "text": "المبلغ المستحق هو 350 ريال."},
    {"id": "ar_022", "category": "numbers", "text": "سيتم التواصل خلال 48 ساعة."},
    {"id": "ar_023", "category": "numbers", "text": "رقم الهوية ينتهي بالرقم 7890."},
    {"id": "ar_024", "category": "numbers", "text": "مدة الاشتراك هي 12 شهراً."},

    # --- Numbers (spoken form) ---
    {"id": "ar_030", "category": "numbers_spoken", "text": "رقم الطلب هو واحد اثنان ثلاثة أربعة خمسة."},
    {"id": "ar_031", "category": "numbers_spoken", "text": "المبلغ ثلاثمائة وخمسون ريالاً."},
    {"id": "ar_032", "category": "numbers_spoken", "text": "سيتم الرد خلال أربع وعشرين ساعة."},

    # --- Dates & Times ---
    {"id": "ar_040", "category": "datetime", "text": "تم تقديم الطلب في الخامس عشر من يناير."},
    {"id": "ar_041", "category": "datetime", "text": "الموعد المحدد هو يوم الثلاثاء الساعة الثالثة مساءً."},
    {"id": "ar_042", "category": "datetime", "text": "وقت العمل من التاسعة صباحاً حتى الخامسة مساءً."},

    # --- Identity / entities ---
    {"id": "ar_050", "category": "identity", "text": "اسمي أحمد الثغافي."},
    {"id": "ar_051", "category": "identity", "text": "البريد الإلكتروني هو ahmed.test@example.com."},
    {"id": "ar_052", "category": "identity", "text": "العنوان هو شارع الملك فهد، رقم 21."},
    {"id": "ar_053", "category": "identity", "text": "الرمز البريدي هو 21577."},

    # --- Mixed Arabic + English ---
    {"id": "ar_060", "category": "mixed", "text": "سيتم إرسال رسالة SMS خلال دقائق."},
    {"id": "ar_061", "category": "mixed", "text": "يرجى إدخال رمز OTP المرسل إلى هاتفك."},
    {"id": "ar_062", "category": "mixed", "text": "يمكنك تحميل التطبيق من Google Play أو App Store."},

    # --- Dialect (Gulf / Saudi-style) ---
    {"id": "ar_070", "category": "dialect", "text": "وش المشكلة اللي تواجهها؟"},
    {"id": "ar_071", "category": "dialect", "text": "خلني أتأكد لك من الطلب."},
    {"id": "ar_072", "category": "dialect", "text": "أبشر، راح أشيك على الموضوع."},
    {"id": "ar_073", "category": "dialect", "text": "لحظة بس أتأكد من النظام."},

    # --- Long realistic sentences ---
    {"id": "ar_080", "category": "long", "text": "استلمت رسالة تفيد بأن حسابي موقوف، ولكن عند تسجيل الدخول يبدو كل شيء طبيعياً، لذلك أود معرفة السبب."},
    {"id": "ar_081", "category": "long", "text": "قبل إجراء أي تعديل، أريد التأكد من أنه لن يتم إضافة أي رسوم إضافية على الفاتورة الشهرية."},
    {"id": "ar_082", "category": "long", "text": "بناءً على البيانات المسجلة لدينا، تم تأكيد الطلب وإرساله إلى قسم الشحن، وسيصلك إشعار فور خروجه من المستودع."},

    # --- Short acknowledgements ---
    {"id": "ar_090", "category": "short", "text": "نعم."},
    {"id": "ar_091", "category": "short", "text": "بالتأكيد."},
    {"id": "ar_092", "category": "short", "text": "لحظة من فضلك."},
    {"id": "ar_093", "category": "short", "text": "تم."},
]
