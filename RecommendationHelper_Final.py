'''
The purpose of this Helper file is to have all the utility functions necessary 
for the recommendation engine so that the main ipynb file can be short and clutter free.
This also provides a chance to decouple the codes so that if anything needs to be modified,
it can be done without having to rerun the main ipynb files.
'''
#importing all required libraries
import pandas as pd
import re
import nltk
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher
import os
import matplotlib.pyplot as plt


'''
The import_dataset function imports the UpdatedResumeDataSet4.csv file
containing the Resumes and the broad category it belings to.
We are randomly generating First Names and Last Names add adding it to the
dataset for the purpose of our analyses.
'''
def import_dataset():
    df_in = pd.read_csv('UpdatedResumeDataSet4.csv' ,encoding='utf-8')
    #generate random names
    from random import shuffle, seed
    from faker.providers.person.en import Provider

    first_names = list(set(Provider.first_names))
    last_names = list(set(Provider.last_names))  ##generates only 473 last names 
    last_names = last_names*3 #so its is multipled 3 times to make more. There maybe some repetitions but its ok
    fn = first_names[0:962] #we need only 962 first names
    ln = last_names[0:962] #we need only 962 last names
    #Add the name columns to the df_in dataframe
    df_in['First_Name']=fn
    df_in['Last_Name']=ln
    return df_in
'''
clean_resume cleans the Resume column of the dataset
This is a text cleaning function that is capable of
removing commonly seen special characters
extra whitespaces,typos,
'''
def clean_resume(text):
    import re
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('#\S+', '', text)  # remove hashtags
    text = re.sub('@\S+', '  ', text)  # remove mentions
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)  # remove punctuations
    text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)  # remove extra whitespace
    text = re.sub('machine learning', 'ML', text)
    text = re.sub('Machine Learning', 'ML', text)
    text = re.sub('Deep Learning', 'DL', text)
    text = re.sub('data', 'Data', text)
    text = re.sub('Hr', 'HR', text)
    text = re.sub('4HANA', 'HANA', text)
    text = re.sub('JAVA','Java',text)
    text = re.sub('Reduce', 'MapReduce', text)
    text = re.sub('TESTING', 'Selenium', text)
    text = re.sub('learnerEducation', 'Junior', text)
    text = re.sub('Exprience', 'experience', text)
    return text

#We were unable to subset or iterate the df_in using the values in the 'Category' column
#The following code will help fix that issue
def fix_subset(df_in):
    skills=[]
    name=[]
    category=[]
    for i in range(0,962):
        category.append(df_in.Category[i])
        skills.append(df_in.cleaned_resume[i])
        fname = df_in.First_Name[i]
        lname = df_in.Last_Name[i]
        name.append(fname+' '+lname)
        df_data = pd.DataFrame.from_records({'Name':name,'Category':category,'Resume':skills})
    return df_data

'''
Compared to using regular expressions on raw text, spaCy’s rule-based matcher engines 
and components not only let you find the words and phrases you’re looking for 
– they also give you access to the tokens within the document and their relationships.
 This means you can easily access and analyze the surrounding tokens, merge spans into single tokens
 or add entries to the named entities 
 source: https://spacy.io/usage/rule-based-matching#matcher
'''
#matching phrases from the keywords using spacy
def match_keywords(skills,keywords):
        phrase_matcher = PhraseMatcher(nlp.vocab,attr="LOWER")
        kw = [nlp.make_doc(skills) for text in keywords ]
        phrase_matcher.add("keywords",None,*kw)
        doc = nlp(skills)
        matches = phrase_matcher(doc)
        found = []
        for match_id,start,end in matches:
            found += doc[start:end]
        return found
'''
Skill_profile function works in tandem with the match_keywords function
Once the PhraseMatcher finds occurances of the input keywords in the resumes,
it returns a list containing the word occurances.

We use this to build a skill profile, by counting the number of times each keyword 
is found in a particular resume. The 'name' is the name of the candidate
extracted from the df_data sataset.
'''   
#build an skill set profile from the matched phrases text
def skill_profile(found,name,keywords):
    #print('building a profile for '+name) #for debugging purposes
    keyword_dict=dict.fromkeys(keywords,0)
    keyword_dict['Name']=name
    #print(keyword_dict)
    for i in found:
        j=str(i)
        if j in keywords:
            keyword_dict[j]+=1
    return keyword_dict

'''
The helper_function() calls the match_keywords and the skill_profile
functions in order to get each candidate's skill profile based on the 
requirement given by the client
'''
def helper_function(df_data,keywords):
    skill_count_list = []
    for i in range(0,len(df_data)):
        skills = df_data.Resume[i]
        name = df_data.Name[i]
        found = match_keywords(skills,keywords)
        skill_count=skill_profile(found,name,keywords)
        skill_count_list.append(skill_count)
    return skill_count_list
'''plot_skillset_profile plots a horizontal stacked barchart 
using the skill set count and names of the candidates to 
help us visually understand which candidate is releavnt to our
requirement
'''
def plot_skillset_profile(df_plot):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10})
    ax = df_plot[:200].plot.barh(title="Resume keywords by category", legend=True, figsize=(20,150), stacked=True)
    labels = []
    for j in df_plot[:200].columns:
        for i in df_plot[:200].index:
            label = str(j)+": " + str(df_plot[:200].loc[i][j])
            labels.append(label)
    patches = ax.patches
    for label, rect in zip(labels, patches):
        width = rect.get_width()
        if width > 0:
            x = rect.get_x()
            y = rect.get_y()
            height = rect.get_height()
            ax.text(x + width/2., y + height/2., label, ha='center',va='center',color ="white",weight='bold')
    return plt.show()

'''
The experience_chunk and the experience_idx functions
attempt to get a gist of the experience profile of any
candidate the client is interested in
'''
def experience_chunks(df_data,name):
    candidate_skills=[]
    for i in range(0,len(df_data)):
        if df_data.Name[i] == name:
            candidate_skills = str(df_data.Resume[i])
    sents = nltk.sent_tokenize(candidate_skills)
    chunk_style = "chunk:{<NNP><NN><NNP><IN><CD><NN.*><NN.*>}"
    parser = nltk.RegexpParser(chunk_style)
    for i in sents:
        words = nltk.word_tokenize(i)
        pos = nltk.pos_tag(words)
        chunks = parser.parse(pos)
    for i in chunks.subtrees():
        if i.label()=="chunk":
            print(i.leaves())

def experience_idx(df_data,name):
    candidate_skills=[]
    exp_profile=[]
    words=[]
    for i in range(0,len(df_data)):
        if df_data.Name[i] == name:
            candidate_skills = str(df_data.Resume[i])
    words = nltk.word_tokenize(candidate_skills)
    for idx,word in enumerate(words):
        if word == 'experience' and words[idx+1]=='Less':
            profile = [words[idx-2],words[idx-1],word,words[idx+1],words[idx+2],words[idx+3],words[idx+4]]
            exp_profile.append(profile)
        elif word == 'experience':
            profile = [words[idx-2],words[idx-1],word,words[idx+1],words[idx+2]]
            exp_profile.append(profile)
    print(exp_profile)

'''
The recommendations functions gives the top 10
candidates of each skill set the client requires
'''
def recommendations(df_skills,keywords):
    candidate_list =[]
    for i in keywords:
        temp_df = df_skills.nlargest(10,i)
        tmp_names = temp_df['Name']
        for name in tmp_names:
            skills_dict = {'Skill':i,'Candidates':name}
            candidate_list.append(skills_dict)
    #df_reco = pd.DataFrame(candidate_list,index=range(0,len(candidate_list)))
    return candidate_list   

'''
the find_common_words functions helps find the buzzwords present in the resumes
This can be very useful for training models,for searching skill sets,understanding
the skillsets in each industry in general
'''
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
import string
def find_common_words(df_resume):
    oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
    totalWords =[]
    Sentences = df_resume['Resume'].values
    cleanedSentences = ""
    lem=WordNetLemmatizer()
    noise=['Details', 'Exprience', 'months', 'company', 'description', '1', 'year','36',
          'Even','offered','frauds','hiding','compared','per','students','web','well',
          'Chat','effective','leadership','Nov','member','along','January','Less', 'Skill','Maharashtra',
          '6','I','project','like', 'India','Pradesh','January','Jan','February','Feb','March','Mar','April','Apr',
          'May','June','Jun','Jul','July','Aug','August','Sep','September','Oct','October','Nov','November',
          'Dec','December','using','monthsCompany','B','C', 'Mumbai', 'Pune', 'Arts', 'A', 'application','24',
          'various', 'Responsibilities','Nagpur','development','Management','Technologies','The',
          'Company','University','British','Designed','Board','new','time','E','May','Ltd','Team',
          'M','Development','etc','Used','2','Council','team','School','Working','work','Developed',
          'Made','given','2016','Sri','required','Learning','Skills','related','involved','3','My',
          '4','Trust','2015','across','This','Lanka','Windows','Adichunchanagiri','Bahadarpur','Gorbanjara',
          'Indira','Priyadarshini','Gandhi','shreekiaspack', '3staragroproducts','luckystationery','ALAMURI',
          'HINDI','Madhya','36','providing','2014','university','board','State','Jalgaon','From','Nashik','Kisan',
          'In','Sr','College','Parola','Dist','www','requirement','com','Higher','State','e','In','used','co','SYSTEM',
          'gives','CURRICULUM','S','OF','LTD','turn','Bengaluru','Karnataka','LifeKonnect','Co','insurance','civil',
          'Aurus','Participated','gathering','meetings','Reviewed','met','February','2006','different','indent',
          '0','S','understanding','writing','Nanded','R','K','KVA','10','28','30','agreed','providing','Timely',
          '2nd','level','Dubai','7','8','e','Helping','300','Su','QATAR','17','5','9','11','12','13','14','15',
          '26','INDIA','5','Thai','27','10','allow','2012','2008','Sr','Pvt','2900s','12D','Asha','2000','2003',
          'ount','Delhi','process','OF','16','30','v10','v11','v12','Pvt','within','9','5','map','Map','Size',
          'used','Unit','9','5','help','also','Inc','yes','June','good','Tech','Like','House','CBSE','Nov','Based',
          '05','07','Asst','To','2010','Pia','Hiralal','Your','2009','2017','Hard','2011','basically','even','P','done',
          'Smt','2004','Apeksha','Naharkar','Thanks','Regards','Talreja','Hindi','Bajaj','Chandrapur','32','alleviate',
          'continued','Savitribai','ANGEL','BOARD','INSTITUTE','Good','Kranti','ountabilities','Thakur','And','P','mumbai','com',
          'Quick','SURYA','kranti','maharastra','PRAKASH','MUMBAI','RAJDEVI','whether', 'Of', 'By', 'ept', 'admin', 'At', 'provide', 
          'As', 'via', 'For', 'With', 'For', 'During', 'On','Of','24x7','201','20','31','7Education','1year','one','800','3000','2D',
          '3D', '3DEducation', '2007', '120','96', '48', '2013', 'Two', '625', '5000', '2000A', '450', '110V', '55', '33', '22','18', 
          '101', '11gEducation', '01','2019', '3years','017', '2018', '20656', '2005']
    for i in range(0,len(df_resume)):
        cleanedText = clean_resume(Sentences[i])
        cleanedSentences += cleanedText
        cleanwords = nltk.word_tokenize(cleanedText)
        requiredWords=[lem.lemmatize(x) for x in cleanwords]
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation and word not in noise:
            totalWords.append(word)
    
    wordfreqdist = nltk.FreqDist(totalWords)
    mostcommon = wordfreqdist.most_common(200)
    return mostcommon

'''
The plot_common_skills functions helps us visualize the 
buzzwords in the resume using a wordcount - word barchart
'''
def plot_common_skills(common_words,category):
    import plotly.express as px    
    skill=[]
    count=[]
    for i in common_words:
        skill.append(i[0])
        count.append(i[1])
    fig = px.histogram(x=skill,y=count, template='simple_white',title= category+' buzzwords')
    fig.update_xaxes(categoryorder='total descending').update_xaxes(title='Buzzwords',tickangle = 45)
    fig.update_yaxes(title='Word count')
    return fig.show()
'''
The experience_rake method extracts import phrases from the candidate's
resume using the RAKE library.
These phrases are then evaluated to see if any words of interest are present.
In our case, we would like to know the experience,skills,trainings etc
'''
import RAKE
import operator
def experience_rake(df_data,name,keywords):
    candidate_skills=[]
    phrases=[]
    for i in range(0,len(df_data)):
        if df_data.Name[i] == name:
            candidate_skills = str(df_data.Resume[i])
    r = RAKE.Rake('SmartStoplist.txt')
    keyphrases = r.run(candidate_skills)
    for phrase,score in keyphrases:
        if score > 3:
            phrases.append(phrase)
    for i in phrases:
        words = i.split(" ")
        for j in keywords:
            if j.lower() in words:
                print(i)



