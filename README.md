# PlanIrregularIDA_Research
The study is  conducted to perform machine learning for the plan irregular building response under the strong ground motion records. IDA was done to study the behaviour.

**WorkFlow**
1. First produce all the definitions, conditions... and all in the STKO Software.
2. The inputs GMS records are extracted from the PEER and then processed by GM_RS_Scaling_Processing.py. The data are needed to be selected and provided in Excelfile in format as provided by PEER search results.
3. The analysis files are then created and processed by STKOPreProcess_Fixedbase.py for various GMs, Scaling factors and Soil conditions for each of the building models opened in STKO.
4. The script files so created are then Analyzed by the Automate_Analyze.py in IDE(like Pycharm, recommended for fast processing) or STKO script terminal, which creates the .mpco files for the analysis.
5. On STKO postprocessor script running terminal Automate_Analyze.py is run to extract the results of superstructure in .txt files in same directory of the script files and .mpco database files.
6. Then the .txt results are transferred and processed by the ResultProcessing.py which accumulates data, sort them, manage in interpretable form of tables and charts.
