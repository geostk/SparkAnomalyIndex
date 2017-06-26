

 Description:
 
 <b>AnomalyIndex:</b>
 *         Calculates the Anomaly Index of all documents belonging to a set of documents
 *         The Anomaly Detection procedure searches for unusual cases based on deviations from the norms of their cluster groups.
 *         The solution is implemented using Apache Spark & Java.
 *
     Usage:
 *         AnomalyIndex <inputfolder> <outputfolder> [K] [DEBUG] 
 *         REQUIRED @<inputfolder> The folder from which the files will be read</inputfolder>
 *         REQUIRED @<outputfolder> The folder where the results will be written to</outputfolder>
 *         OPTIONAL @[K] : Write the top K  the results (Descending) , default=5
 *         OPTIONAL @[DEBUG] : Write intermediate results to output folder
 *
 