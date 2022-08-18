from html import entities
import nlpcloud

with open("Name-Entity-Recognition/api.txt", "r") as f:
    api_key = f.read()

client = nlpcloud.Client("en_core_web_lg",'77a687fa0226362e6ab53549683cbabcaf8d2947',gpu =False,lang="en")
entities = client.entities("funny, I signed up thinking free, but then you spring the 14 day trial. You wasted 30 mins of my time today\r\n \r\nremove and delete my account at once. And stop misleading people!!!\r\n \r\n--\r\nCorrina Graco\r\nVP of Marketing | Google Ads for Funeral Homes\r\nUK 01638 474213 | USA 855-534-3713 \r\nMildenhall UK | Sarasota FL  \r\nwww.funeralfusion.co.uk | www.funeralfusion.com\r\n""",searched_entity="ORG")
print(entities)