#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df=pd.read_csv('ab_data.csv')
df.head()


# 
# b. Use the cell below to find the number of rows in the dataset.

# In[3]:


df.shape


# 
# 
# 
# c. The number of unique users in the dataset.

# In[4]:


df.nunique()


# 
# 
# 
# 
# d. The proportion of users converted.

# In[5]:


df.converted= df.converted.map({True: 1, False: 0})
df.groupby(['user_id'])['converted'].agg({'converted':'sum'}).mean()


# e. The number of times the `new_page` and `treatment` don't match.

# In[6]:



unwanted_rows = df[(df['group'] == 'treatment') != (df['landing_page'] == 'new_page')]
unwanted_rows.shape


# f. Do any of the rows have missing values?

# In[7]:


np.count_nonzero(df.isnull().values)   


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[8]:


unwanted_index = unwanted_rows.index
df2= df.drop(unwanted_index)
df2.head()


# In[9]:


df2.shape


# In[10]:



# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[11]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[12]:


df2[df2['user_id'].duplicated()]


# 
# 
# c. What is the row information for the repeat **user_id**? 

# In[13]:


df2[df2['user_id'].duplicated()].info()


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[14]:


df2.drop([2893],inplace=True)


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[15]:


df2.info()


# In[16]:


total=len(df2['converted'])
total


# In[17]:



converted=df2[df2['converted']==1]
converted.head()


# In[18]:



p_converted=len(converted)/total
p_converted


# b. Given that an individual was in the `control` group, what is the probability they converted?

# p(control&converted)/p(control) this is the equation to work with

# In[19]:


len(df2['converted'])


# In[20]:


df2.groupby(["group", "converted"]).size().reset_index(name="group and converted")


# (control & converted)/total control=17489/145274=.1203863

# \c. Given that an individual was in the `treatment` group, what is the probability they converted?

# p(treatment&converted)/p(treatment) this is the equation to work with

# In[21]:


17264/145310


# d. What is the probability that an individual received the new page?

# In[22]:


df2['landing_page'].value_counts(normalize=True)


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# There is insufficient evidence to draw from to conclude that new treatment page leads to more conversions. According to the data, the probility of converting was higher in the control group than in the treatment group. Additionally, half of the individuals received the new page. 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

#  null = 𝑝𝑜𝑙𝑑 -  𝑝𝑛𝑒𝑤 ≥	0.05 
#  alternative = 𝑝𝑜𝑙𝑑 -  𝑝𝑛𝑒𝑤 < 0.05

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[23]:


p_new= .11880806551510564


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[24]:


p_old= .1203863


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[25]:


df2.groupby('group').size()


# d. What is $n_{old}$, the number of individuals in the control group?

# In[26]:


n_new=145310


# In[27]:


n_old=145274


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[28]:


new_page_converted = np.random.choice([0, 1], size=n_new, p=[1-p_new, p_new]) 


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[30]:


old_page_converted = np.random.choice([0, 1], size=n_old, p=[1-p_old, p_old])


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[31]:


p_diff=new_page_converted.mean() - old_page_converted.mean()
p_diff


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[32]:



sample_data=df2.sample(10000)


# In[33]:


p_diffs= []
for _ in range(10000):
    new_page_converted = np.random.binomial(n_new,p_new)
    old_page_converted = np.random.binomial(n_old, p_old)
    diff = new_page_converted/n_new - old_page_converted/n_old
    p_diffs.append(diff)
   


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[34]:


plt.hist(p_diffs);


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[35]:


(p_diffs > p_diff).mean()


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# The value above is the simulation P-value which is the probability of getting our statistic or a more extreme value if the null is true. If the p-value is greater than the threshold of .05 (alpha), we are more likely to choose the null hypothesis because null = 𝑝𝑜𝑙𝑑 - 𝑝𝑛𝑒𝑤 ≥ 0.05. Thus, there is no significant difference between choosing the new and the old pages. We would not reject the null hypothesis in this case because the p-value is greater than alpha (.05).

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[36]:


df2.groupby(["landing_page", "converted"]).size().reset_index(name="landing_page and converted")


# In[37]:


import statsmodels.api as sm
from scipy.stats import norm
convert_old = 17489
convert_new = 17264
n_old = 145274
n_new = 145310


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[38]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new ], [n_old, n_new])


# In[39]:


norm.cdf(z_score)


# In[40]:


z_score


# In[41]:


p_value


# In[42]:


norm.ppf(1-(0.05/2))


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# The z-score represents how many standard deviations above or below the mean a value is; in this case it is +1.31 standard deviations away from the mean. Since alpha is .05, the confidence interval is .95 or 95% (1-.05). Since the z-score is between the confidence interals of -1.96 and +1.96 standard deviations and the p-value is also greater than .05, we cannot reject the null hypothesis.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# We will be performing logistic regression because we are only predicting two possible outcomes.

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[86]:


# import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
df2['intercept'] = 1
df2['ab_page'] = (df2.group == 'treatment').astype(int)
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[103]:


regression_model = sm.Logit(df2.converted, df2[['intercept', 'ab_page']])


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[104]:


final=regression_model.fit()
final.summary2()


# In[89]:


np.exp(-0.0150)


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# THe p-values are the same for part 2 and 3 ( .1899).

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# Considering other vairables that may be categorical or quantitative may provide more accuracy to find the most signigicant factors that cause an individual to convert. However, including additional variables and dimensions, decreases the viable sample size and thus leads to overspecification and lack of a general solution/conclusion. Overfitting may occur with the addition of more terms into our regression model and while the model may be more "perfect", it is only "perfect" for the particular dataset used and would not provide a good overall idea.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[112]:


np.exp(0.0057),np.exp(-0.0118)


# The above are exponentials of the coefficients for US and CA respectively (obtained from logitmodel below. The values show that country did not have a significant impact on conversion as US individuals were 1.005% more likely to convert than not, and CA individuals were .988% more likely to convert than not.

# In[90]:


countries_df=pd.read_csv('countries.csv')
countries_df.head()


# In[91]:


new_countries=countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
new_countries.head()


# In[92]:


# new_countries['country'].value_counts()


# In[93]:


new_countries['US'] = (new_countries.country == 'US').astype(int)
new_countries['CA'] = (new_countries.country == 'CA').astype(int)
new_countries['UK'] = (new_countries.country == 'UK').astype(int)
new_countries.head()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[100]:


# interaction=new_countries["country"] * new_countries["ab_page"]
# interaction.head()


# In[106]:


new_countries['US_ab_page'] = new_countries['US']*new_countries['ab_page']
new_countries['CA_ab_page'] = new_countries['CA']*new_countries['ab_page']
logit_model = sm.Logit(new_countries['converted'], new_countries[['intercept', 'ab_page','US', 'CA', 'US_ab_page', 'CA_ab_page']])
final_lm=logit_model.fit()


# In[107]:


final_lm.summary2()


# In[111]:


np.exp(-0.0314),np.exp(-0.0783)


# The summary shows that an individual lives in the US and converts is .969% more likely to convert than not. Similarly, an individual in CA converting is .925% more likely to convert than not. This shows that interaction between country and pages don't have a significant impact on conversion rates. Based on the data, I would continue using the old page.

# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[51]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




