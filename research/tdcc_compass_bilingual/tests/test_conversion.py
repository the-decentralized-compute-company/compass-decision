import unittest
from prepare_extra import massive_families,massive_quality,image_family,slot_template
from schemas import massive_questions,nli_questions,INTENT_IDS

class ConversionTests(unittest.TestCase):
 def test_sixty_fixed_bilingual_choices(self):
  self.assertEqual(len(INTENT_IDS),60)
  self.assertEqual(set(massive_questions('tr')['intent']['criteria']),set(massive_questions('en')['intent']['criteria']))
  self.assertEqual(len(set(INTENT_IDS.values())),60)
 def test_slot_values_removed_names_retained(self):
  self.assertEqual(slot_template('Play [song_name : Song A] please'),'play [song_name] please')
  self.assertEqual(slot_template('PLAY [song_name : Song B] please'),'play [song_name] please')
  self.assertNotEqual(slot_template('Play [song_name : A]'),slot_template('Play [artist_name : A]'))
 def test_cross_language_template_components(self):
  def row(id,text,annot,part):return {'id':id,'utt':text,'annot_utt':annot,'partition':part,'scenario':'play'}
  locales={'en':{'a':row('a','play red','play [song : red]','train'),'b':row('b','play blue','play [song : blue]','dev'),'c':row('c','other command','other command','test')},'tr':{'a':row('a','kırmızı çal','[song : kırmızı] çal','train'),'b':row('b','mavi çal','[song : mavi] çal','dev'),'c':row('c','yeşil çal','[song : yeşil] çal','test')}}
  by_id,families=massive_families(locales)
  self.assertEqual(by_id['a'],by_id['c'])
  self.assertEqual(set(families[by_id['a']]['partitions']),{'train','dev','test'})
 def test_image_not_just_caption_group(self):
  self.assertEqual(image_family({'captionID':'123.jpg#1'}),image_family({'captionID':'123.jpg#9'}))
 def test_quality_filter_does_not_invent_labels(self):
  yes={'intent_score':1,'grammar_score':4};bad={'intent_score':0,'grammar_score':2}
  self.assertTrue(massive_quality({'judgments':[yes,yes,bad]}))
  self.assertFalse(massive_quality({'judgments':[yes,bad,bad]}))
 def test_nli_label_sets_identical(self):
  self.assertEqual(set(nli_questions('en')['relation']['criteria']),{'entailment','contradiction','neutral'})
  self.assertEqual(set(nli_questions('tr')['relation']['criteria']),set(nli_questions('en')['relation']['criteria']))
if __name__=='__main__':unittest.main()
