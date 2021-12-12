# ECON860_final
My final for Clemson's ECON 8600, Data Analysis for Economics

**Answers**
1c. I found 4 factors was optimal.

1d. I utilized KMeans and KMedoids clustering

1e. For both, 3 clusters was optimal

1f: It is unclear. I can compare my 3-cluster results directly for KMedoids and KMeans. I find that my cluster with KMeans has a better value than KMedoids for Silhouette Score, but a worse value for SSD. Therefore, it is not clear which algorithm works better.

1g. Yes. I analyzed answers from GB. My answers for 1c-1e were the same. 1f differed though, as my 3-cluster result using KMedoids outperformed my KMeans results in Silhouette Score and SSD. 

I annotate my answers through comments in my python files as well.

**Instructions**
1. Download the repository to a folder on your computer.

2. Open your Terminal/Command Prompt

3. Navigate to the folder you downloaded the repostiory
    
    ex. cd Desktop --> cd ECON8600 --> cd final8600

4. Run the first python file with the following command:
    
    python final.py
    
    This will create several png files in your folder that will let you visualize the data. There should be outputs using KMeans and KMedoids               clustering, with outputs showing results from 1-5 clusters. There will also be a file called "plot.png" which helps you determine how many factors     to use; there will be lines in the terminal and comments in the code that help you determine the optimal number of factors as well. I chose 4.

5. Run the second python file with the following command:
    
    python finalgb.py
    
    This allows you to answer 1g, by comparing the results from all countries with the output from just GB. It will output all the same png's as before     with "GB" included to distinguish them. You should find the data is similar but distinctly different from your first outputs.
    
This completes the instructions. The code can be edited if you desire to see outputs from other countries by changing "GB" in line 24 of finalgb.py to your desired country. Other tweaks can be made as well, but this code satisfies 1c-1g.
