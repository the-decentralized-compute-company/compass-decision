"""Authored schema descriptions. Dataset labels are preserved, not re-annotated."""
MASSIVE_DESCRIPTIONS = {
'alarm_query':('List alarms','Alarmları listele'),
'alarm_remove':('Remove an alarm','Alarmı sil'),
'alarm_set':('Set an alarm','Alarm kur'),
'audio_volume_down':('Lower volume','Sesi azalt'),
'audio_volume_mute':('Mute audio','Sesi kapat'),
'audio_volume_other':('Set or check volume','Sesi ayarla veya sorgula'),
'audio_volume_up':('Raise volume','Sesi artır'),
'calendar_query':('Check calendar','Takvimi sorgula'),
'calendar_remove':('Remove calendar event','Takvim etkinliğini sil'),
'calendar_set':('Add calendar event','Takvim etkinliği ekle'),
'cooking_query':('Ask cooking question','Yemek pişirme sorusu sor'),
'cooking_recipe':('Find a recipe','Yemek tarifi bul'),
'datetime_convert':('Convert time zone','Saat dilimini dönüştür'),
'datetime_query':('Ask date or time','Tarih veya saati sor'),
'email_addcontact':('Add email contact','E-posta kişisi ekle'),
'email_query':('Check email','E-postaları sorgula'),
'email_querycontact':('Find contact details','Kişi bilgilerini bul'),
'email_sendemail':('Send email','E-posta gönder'),
'general_greet':('Exchange greetings','Selamlaş'),
'general_joke':('Tell a joke','Şaka anlat'),
'general_quirky':('Casual assistant chat','Asistanla günlük sohbet et'),
'iot_cleaning':('Control cleaning device','Temizlik cihazını yönet'),
'iot_coffee':('Make coffee','Kahve hazırla'),
'iot_hue_lightchange':('Change light color','Işığın rengini değiştir'),
'iot_hue_lightdim':('Dim lights','Işıkları kıs'),
'iot_hue_lightoff':('Turn lights off','Işıkları kapat'),
'iot_hue_lighton':('Turn lights on','Işıkları aç'),
'iot_hue_lightup':('Brighten lights','Işıkları artır'),
'iot_wemo_off':('Turn smart plug off','Akıllı prizi kapat'),
'iot_wemo_on':('Turn smart plug on','Akıllı prizi aç'),
'lists_createoradd':('Create or extend list','Liste oluştur veya ekle'),
'lists_query':('Check a list','Listeyi sorgula'),
'lists_remove':('Remove from list','Listeden sil'),
'music_dislikeness':('Dislike music','Müziği beğenme'),
'music_likeness':('Like music','Müziği beğen'),
'music_query':('Ask about music','Müzik hakkında sor'),
'music_settings':('Change music settings','Müzik ayarlarını değiştir'),
'news_query':('Ask for news','Haberleri sor'),
'play_audiobook':('Play an audiobook','Sesli kitap aç'),
'play_game':('Play a game','Oyun oyna'),
'play_music':('Play music','Müzik çal'),
'play_podcasts':('Play a podcast','Podcast aç'),
'play_radio':('Play radio','Radyo aç'),
'qa_currency':('Ask exchange rate','Döviz kurunu sor'),
'qa_definition':('Define a term','Terimi tanımla'),
'qa_factoid':('Ask factual question','Olgusal bilgi sor'),
'qa_maths':('Calculate arithmetic','Aritmetik hesapla'),
'qa_stock':('Ask stock price','Hisse fiyatını sor'),
'recommendation_events':('Recommend events','Etkinlik öner'),
'recommendation_locations':('Recommend places','Mekân öner'),
'recommendation_movies':('Recommend movies','Film öner'),
'social_post':('Post to social media','Sosyal medyada paylaş'),
'social_query':('Check social media','Sosyal medyayı sorgula'),
 'takeaway_order':('Order takeaway food','Paket yemek sipariş et'),
 'takeaway_query':('Find takeaway options','Paket yemek seçeneklerini sor'),
 'transport_query':('Ask transport schedule','Ulaşım seferlerini sor'),
 'transport_taxi':('Request a taxi','Taksi iste'),
 'transport_ticket':('Book transport ticket','Ulaşım bileti ayırt'),
 'transport_traffic':('Ask traffic conditions','Trafik durumunu sor'),
 'weather_query':('Ask weather forecast','Hava durumunu sor'),
}
INTENT_IDS={intent:str(n) for n,intent in enumerate(sorted(MASSIVE_DESCRIPTIONS))}

def massive_questions(language):
    i=0 if language=='en' else 1
    return {'intent':{'type':'choice','instructions':('Classify the speaker’s intent. Select the matching request; do not perform it.' if i==0 else 'Konuşanın niyetini sınıflandır. Uygun isteği seç; isteği gerçekleştirme.'), 'criteria':{INTENT_IDS[k]:v[i] for k,v in sorted(MASSIVE_DESCRIPTIONS.items())}}}

def nli_questions(language):
    if language=='en':
        ins='Assume the premise is true. Does it entail, contradict, or leave the hypothesis undetermined? Use only the premise.'
        criteria={'entailment':'The hypothesis must be true.','contradiction':'The hypothesis must be false.','neutral':'The hypothesis may be true or false.'}
    else:
        ins='Öncülün doğru olduğunu varsay. Önermeyi destekliyor mu, onunla çelişiyor mu, yoksa belirsiz mi bırakıyor? Yalnızca öncülü kullan.'
        criteria={'entailment':'Önerme kesinlikle doğrudur.','contradiction':'Önerme kesinlikle yanlıştır.','neutral':'Önerme doğru da yanlış da olabilir.'}
    return {'relation':{'type':'choice','instructions':ins,'criteria':criteria}}
