
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
import os

# CSS ilə dizayn tənzimləmələri
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        color: #2E8B57;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #555555;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-box {
        background-color: #E6F2F0;
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
    }
    .center-button {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .small-image {
        max-width: 300px;
        margin: auto;
        display: block;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(r"C:\Users\user\Desktop\AI PROJECTS\Technofest Project\my_model.keras")

@st.cache_data
def load_class_labels():
    with open(r"C:\Users\user\Desktop\AI PROJECTS\Technofest Project\class_labels.json", "r") as f:
        return json.load(f)

model = load_model()
class_labels = load_class_labels()

solutions = {
    "Apple Black rot": """Bu xəstəlik meyvələrdə qara ləkələr və çürümələrə səbəb olur, ona görə də zədələnmiş meyvələri vaxtında yığmaq və atmaq vacibdir. Meyvələrin düzgün saxlanması xəstəliyin yayılmasının qarşısını almağa kömək edir. Bundan əlavə, sağlam bitkiləri qorumaq üçün müntəzəm yoxlama və təmizləmə işləri aparılmalıdır.""",

    "Apple Healthy": """Heç bir problem yoxdur. Bitkinizin sağlamlığını qorumaq üçün normal qaydada baxım edin və xəstəliklərin qarşısını almaq üçün vaxtaşırı yoxlama aparın. Sağlam şərtlərdə suvarma və gübrələmə davam etdirilməlidir.""",

    "Apple Scab": """Fungisid tətbiq etmək xəstəliyin qarşısını almaq üçün vacibdir. Nəmli mühitdən uzaq durmaq və xəstə yarpaqları təmizləmək xəstəliyin yayılmasının qarşısını alır. Bağda təmizliyi müntəzəm saxlamaq tövsiyə olunur.""",

    "Cedar apple rust": """Fungisid müalicəsi ilə xəstəlikdən qorunmaq mümkündür. Təsirlənmiş hissələri dərhal kəsib atmaq vacibdir. Bitkinin yaxınlığında xəstəlik daşıyan ağaclardan uzaq durmaq faydalıdır.""",

    "Bell pepper Bacterial spot": """Xəstə bitkiləri tez bir zamanda təcrid edin ki, xəstəlik yayılmasın. Müvafiq fungisid və pestisidlərdən istifadə edin. Bitkinin ətrafını təmiz və quru saxlayın.""",

    "Bell pepper Healthy": """Heç bir problem yoxdur. Bitkinizin sağlam qalması üçün normal baxım işlərini davam etdirin. Vaxtaşırı zərərvericilərə qarşı yoxlama aparmaq faydalıdır.""",

    "Cherry Healthy": """Bitkiniz sağlamdır. Müntəzəm suvarma və gübrələmə ilə sağlamlığını qoruyun. Xəstəliklərin qarşısını almaq üçün profilaktik tədbirlər görmək məsləhətdir.""",

    "Cherry Powdery mildew": """Toz küf xəstəliyi üçün uyğun fungisid tətbiq edin. Bitkini günəşli və yaxşı havalandırılan yerdə saxlayın. Xəstə yarpaqları vaxtında təmizləyin və atın.""",

    "Corn Common rust": """Xəstəlikdən qorunmaq üçün müvafiq fungisidlərdən istifadə edin. Xəstə yarpaqları dərhal təmizləyin. Əkin sahəsini təmiz və havalandırılan vəziyyətdə saxlayın.""",

    "Corn Gray leaf spot": """Xəstə yarpaqları vaxtında təmizləyin və atın ki, xəstəlik yayılmasın. Fungisid tətbiq etməyi unutmayın. Havanı yaxşı sirkulyasiya edən şərait yaradın.""",

    "Corn Healthy": """Bitkiniz sağlamdır. Normal baxımı davam etdirin və xəstəlik əlamətlərini izləyin. Vaxtaşırı zərərvericilərə qarşı mübarizə aparın.""",

    "Corn Northern Leaf Blight": """Müvafiq fungisid və xəstəlikdən davamlı toxumlardan istifadə edin. Zədələnmiş hissələri dərhal kəsib atın. Əkin sahəsində təmizliyi təmin edin.""",

    "Grape Black Measles": """Fungisid tətbiq edin və suvarmanı düzgün tənzimləyin ki, xəstəlik yayılmasın. Zədələnmiş meyvə və yarpaqları toplayıb məhv edin. Bağda müntəzəm təmizliyi təmin edin.""",

    "Grape Black rot": """Xəstə yarpaqları və meyvələri vaxtında təmizləyin və atın. Fungisid tətbiq etməyi unutmayın. Suvarmanı balanslı aparın və nəmi azaldın.""",

    "Grape Healthy": """Heç bir problem yoxdur. Bağınızı sağlam saxlamaq üçün normal baxım işlərini davam etdirin. Profilaktik tədbirlər görmək faydalıdır.""",

    "Grape Isariopsis Leaf Spot": """Fungisid tətbiq edin və təsirlənmiş yarpaqları çıxarın ki, xəstəlik yayılmasın. Bağı təmiz və havalandırılan saxlayın. Zədələnmiş hissələri vaxtında məhv edin.""",

    "Peach Bacterial spot": """Mis tərkibli fungisid istifadə edin və xəstə bitkiləri təcrid edin. Bağda təmizliyi təmin edin və zədələnmiş hissələri kəsib atın. Suvarmanı müntəzəm və balanslı aparın.""",

    "Peach Healthy": """Heç bir problem yoxdur. Bitkinizin sağlamlığını qorumaq üçün normal baxımı davam etdirin. Vaxtaşırı yoxlama apararaq xəstəliklərin qarşısını alın.""",

    "Potato Early blight": """Xəstəlikdən davamlı toxumlar istifadə edin və uyğun fungisid tətbiq edin. Zədələnmiş yarpaqları vaxtında təmizləyin. Əkin sahəsinin təmizliyinə diqqət yetirin.""",

    "Potato Healthy": """Bitkiniz sağlamdır. Normal suvarma və gübrələmə ilə sağlamlığı qoruyun. Xəstəliklərin qarşısını almaq üçün profilaktik tədbirlər görmək lazımdır.""",

    "Potato Late blight": """Xəstəlikdən davamlı toxumlar və fungisid tətbiq etmək xəstəliyin yayılmasını azaldır. Zədələnmiş hissələri vaxtında kəsib atın. Bağda müntəzəm təmizlik aparın.""",

    "Strawberry Healthy": """Heç bir problem yoxdur. Bitkinin sağlamlığını qorumaq üçün baxımı davam etdirin. Vaxtaşırı zərərvericilər və xəstəliklər üçün yoxlama aparın.""",

    "Strawberry Leaf scorch": """Fungisid tətbiq edin və təsirlənmiş yarpaqları təmizləyin. Bağı yaxşı havalandırın və nəmi azaldın. Zədələnmiş hissələri kəsib məhv edin.""",

    "Tomato Bacterial spot": """Mis tərkibli fungisid istifadə edin və xəstə bitkiləri dərhal təcrid edin. Bağda təmizliyi təmin edin. Zərərvericilərlə mübarizə aparın.""",

    "Tomato Early blight": """Fungisid tətbiq edin və təsirlənmiş yarpaqları təmizləyin. Bitkini yaxşı havalandırılan yerdə saxlayın. Zədələnmiş hissələri vaxtında kəsib atın.""",

    "Tomato Healthy": """Bitkiniz sağlamdır. Normal baxımı davam etdirin və xəstəliklərə qarşı profilaktik tədbirlər görün. Vaxtaşırı yoxlama aparmaq faydalıdır.""",

    "Tomato Late blight": """Xəstəlikdən davamlı toxumlar və uyğun fungisid tətbiq edin. Zədələnmiş hissələri dərhal kəsib atın. Bağda təmizliyi təmin edin və nəmi azaldın.""",

    "Tomato Leaf Mold": """Havalandırmanı yaxşılaşdırın və fungisid istifadə edin. Təsirlənmiş yarpaqları təmizləyin və bağda təmizliyi saxlayın. Suvarmanı balanslı aparın.""",

    "Tomato Mosaic virus": """Xəstə bitkiləri dərhal çıxarın və alətləri dezinfeksiya edin. Virusun yayılmasının qarşısını almaq üçün gigiyena qaydalarına əməl edin. Sağlam bitkiləri qorumaq üçün müntəzəm yoxlama aparın.""",

    "Tomato Septoria leaf spot": """Xəstə yarpaqları çıxarın və fungisid tətbiq edin. Bağı havalandırılan saxlayın. Zədələnmiş hissələri vaxtında məhv edin.""",

    "Tomato Spider mites": """Zərərvericilərə qarşı uyğun pestisid tətbiq edin. Bağda nəm səviyyəsini nəzarətdə saxlayın. Müntəzəm yoxlama apararaq erkən müdaxilə edin.""",

    "Tomato Target Spot": """Fungisid istifadə edin və təsirlənmiş yarpaqları çıxarın. Bağı təmiz saxlayın və havalandırmanı təmin edin. Zədələnmiş hissələri dərhal məhv edin.""",

    "Tomato Yellow Leaf Curl Virus": """Xəstə bitkiləri çıxarın və virus daşıyıcı həşəratları məhv edin. Gigiyena qaydalarına əməl edin və alətləri dezinfeksiya edin. Sağlam bitkilərin mühafizəsinə diqqət yetirin."""
}

# Sol panel
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/415/415733.png", width=150)
st.sidebar.title("Bitki Xəstəlikləri Aşkarlanması")
st.sidebar.write("Bu tətbiq şəkilləri tanıyaraq bitki xəstəliklərini müəyyən edir və həllər təqdim edir.")

# Başlıq və alt başlıq
st.markdown('<div class="main-title">Bitki Xəstəliklərinin Aşkarlanması</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Şəkil yükləyin və xəstəliyi dəqiq müəyyən edin</div>', unsafe_allow_html=True)

# Şəkil yükləmə
uploaded_file = st.file_uploader("Şəkil yüklə (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(file_path, caption="Yüklənmiş şəkil", width=250)  # Şəkil eni 250 piksel olacaq



    if st.button("Proqnoz et"):
        image = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        class_name = class_labels.get(str(class_index), "Naməlum")

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.success(f"Nəticə: **{class_name}**")
        st.info(solutions.get(class_name, "Bu xəstəlik üçün həll tapılmadı."))
        st.markdown('</div>', unsafe_allow_html=True)

    # Faylı silmək (təhlükəsizlik məqsədilə)
    if os.path.exists(file_path):
        os.remove(file_path)


