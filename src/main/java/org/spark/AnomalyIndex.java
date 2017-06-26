package org.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;

import java.text.Normalizer;
import java.util.*;
import java.io.Serializable;

/**
 * Description:
 *
 *    AnomalyIndex:
 *         Calculates the Anomaly Index of all documents belonging to a set of documents
 *         The Anomaly Detection procedure searches for unusual cases based on deviations from the norms of their cluster groups.
 *         The solution is implemented using Apache Spark & Java.
 *
 *    Usage:
 *         AnomalyIndex <inputfolder> <outputfolder> [K] [DEBUG]
 *         REQUIRED @<inputfolder> The folder from which the files will be read</inputfolder>
 *         REQUIRED @<outputfolder> The folder where the results will be written to</outputfolder>
 *         OPTIONAL @[K] : Write the top K  the results (Descending) , default=5
 *         OPTIONAL @[DEBUG] : Write intermediate results to output folder
 *
 */
public class AnomalyIndex {

    public static void main(String[] args) throws Exception {

        //get the execution parameters of the program
        ParamsExtractor paramsExtractor = new ParamsExtractor(args).invoke();
        String inputPath = paramsExtractor.getInputPath()+"/*/";
        Integer K  = paramsExtractor.getK();
        String outputPath = paramsExtractor.getOutputPath();
        Boolean DEBUG = paramsExtractor.getDebug();

        // get the context object
        JavaSparkContext sc = getSparkContext();

        //Read the input directory of text files into a PairRDD with the key being the file name
        JavaPairRDD<String,String> files = sc.wholeTextFiles(inputPath);

        //Create a PairRDD with the key being the file name and the value the word
        JavaPairRDD <String,String>  FileAndWord  =files.flatMapToPair( s-> {

                List <Tuple2<String,String>> mylist= new ArrayList<>();
                String file_name = s._1();

                String content= Normalizer.normalize(s._2(),Normalizer.Form.NFD);
                content=content.replaceAll("[^\\p{ASCII}]",""); //remove non ASCII characters

                String [] words=content.split("[ \t\n\r]+"); //split to words
            for (String word : words) {
                mylist.add(new Tuple2<>(file_name, word));
            }

                return mylist.iterator();
        });

        FileAndWord.cache(); // cache it since we are going to to use it again

        final long totalWordsCount = FileAndWord.count(); //total count of words in all files

        //Initialize word count with 1 for each occurrence.
        //Name the variable twos , so we can just make a difference...
        JavaPairRDD<Tuple2<String,String>, Integer> twos = FileAndWord.mapToPair((Tuple2<String,String> s) ->  new Tuple2<>(s, 1) );

        // find the total count for each unique word per file
        JavaPairRDD<Tuple2<String,String>, Integer> counts = twos.reduceByKey((Integer i1, Integer i2) -> i1 + i2);

        // counts: { [(file1, word1), f1], [(file2, word2), f2], ... }
        // create another RDD as:
        // fileAsKey: { [file1, (f1, word1)], [file2, (f2, word2)], ...]
        JavaPairRDD<String, Tuple2<Integer,String>> fileAsKey =
                counts.mapToPair((Tuple2<Tuple2<String,String>,Integer>  s) -> {
                    String fileID = s._1._1;
                    String word = s._1._2;
                    Integer frequency = s._2;
                    Tuple2<Integer, String> freqAndWord = new Tuple2<>(frequency, word);
                    return new Tuple2<>(fileID, freqAndWord);
                });

        // now group by file <file , Iterable<occurence of word in file , word> >
        JavaPairRDD<String, Iterable<Tuple2<Integer,String>>> FrequencyListGroupedByFile = fileAsKey.groupByKey();

        // build an associative array to be used for finding word count in all the documents (word , count)
        // Broadcast the key-value pairs in this RDD as a Map.
        JavaPairRDD<String,Integer> wordOccurrenceTotal =  FileAndWord.mapToPair(x-> new Tuple2<>(x._2(),1)).reduceByKey( (x,y) -> x+ y );
        //Broadcast< Map<String, Integer>> wordOccurrenceTotalMap = sc.broadcast(wordOccurrenceTotal.collectAsMap());
        Map<String, Integer> map1 = new HashMap<>();
        map1.putAll(wordOccurrenceTotal.collectAsMap());
        Broadcast< Map<String, Integer>> wordOccurrenceTotalMap = sc.broadcast(map1);


        // build an associative array to be used for finding word count per file (file , count)
        // Broadcast the key-value pairs in this RDD as a Map.
        JavaPairRDD<String,Integer> NumWordsPerFile = FileAndWord.mapToPair(x-> new Tuple2<>(x._1(),1)).reduceByKey((x, y) -> x + y);
        //Broadcast<Map<String, Integer>> NumWordsPerFileMap = sc.broadcast(NumWordsPerFile.collectAsMap());
        Map<String, Integer> map2 = new HashMap<>();
        map2.putAll(NumWordsPerFile.collectAsMap());
        Broadcast< Map<String, Integer>> NumWordsPerFileMap = sc.broadcast(map2);


        // It is time , to calculate the AnomalyIndex Score for each document
        // Let's do it
        JavaPairRDD<String, Double> anomalyIndex = FrequencyListGroupedByFile // <s._1 = file , s._2 = Iterable<occurence of word in file , word> >
                .mapToPair(s -> {
                    double anomalyScore=0.0;

                    String fileName = s._1;
                    Integer numWordsInFile = NumWordsPerFileMap.value().get(fileName);

                    //Iterate through the list of words for each document and make the calculations
                    for (Tuple2<Integer,String> n : s._2) {  //n = <n._1 = occurence of word in file , n._2 = word>
                        Integer occurrenceWordInFile = n._1();
                        Integer occurrenceWordTotal =  wordOccurrenceTotalMap.value().get(n._2());

                        double  wordInFileFrequency = (double) occurrenceWordInFile / numWordsInFile;
                        double  wordInAllFilesFrequency = (double) occurrenceWordTotal / totalWordsCount;

                       anomalyScore+= wordInFileFrequency * java.lang.Math.log(wordInFileFrequency /wordInAllFilesFrequency );
                    }
                    return new Tuple2<>(fileName, anomalyScore);
                });

            //take K and sort by value descending , then convert to RDD
            List<Tuple2<String,Double>> topK = anomalyIndex.takeOrdered(K, AnomalyIndexComparatorDescending.INSTANCE);
            JavaPairRDD<String,Double> topKResults = sc.parallelizePairs(topK);

            //write results to output
            topKResults.saveAsTextFile(outputPath+"_Results");

        //Save intermediate results for debugging
        if (DEBUG) {
            anomalyIndex.saveAsTextFile(outputPath+"_DBG_anomalyIndex");
            wordOccurrenceTotal.saveAsTextFile(outputPath + "_DBG_WordOccurenceTotal");
            NumWordsPerFile.saveAsTextFile(outputPath + "_DBG_NumWordsPerFile");
            FrequencyListGroupedByFile.saveAsTextFile(outputPath + "_DBG_GroupedByFile");
        }

        sc.stop(); //bye bye

    } //end main

    /**
     *
     * @return an instance of JavaSparkContext , configured with optimized settings
     */
    private static JavaSparkContext getSparkContext() {

         // Create a spark Config and Add parameters for efficiency
        SparkConf sparkConf = new SparkConf().setAppName("AnomalyIndex").setMaster("local");

        //increase the size  of the in-memory buffer for each shuffle file output stream.
        sparkConf.set("spark.shuffle.file.buffer","64k").set("spark.kryoserializer.buffer","24m");

        //change JAVA VM options to use GCGC Garbage Collector
        sparkConf.set("spark.executor.extraJavaOptions", "-XX:+UseG1GC");

        //change compression codec to lz4
        sparkConf.set("spark.io.compression.codec", "lz4");

        //change JAVA VM options to make pointers be four bytes instead of eight.
        sparkConf.set("spark.executor.extraJavaOptions", "-XX:+UseCompressedOops");

        //change to Kryo serializer #see http://spark.apache.org/docs/latest/tuning.html#data-serialization
        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");

        //Adjust memory allocation for shuffling
        //sparkConf.set("spark.memory.fraction", "0.4");
        //sparkConf.set("spark.memory.storageFraction", "0.7");

        JavaSparkContext context = new JavaSparkContext(sparkConf); // create a context object
        return context;
    }

    /**
     * Defines a Comparator for Tuple2<String,Double>
     * Implements java.io.Serializable,as it will  be used as ordering method in SPARK\s serializable data structures
     */
    private static class AnomalyIndexComparatorDescending implements Comparator<Tuple2<String,Double>>, Serializable {
        final static AnomalyIndexComparatorDescending INSTANCE = new AnomalyIndexComparatorDescending();
        // sort descending based on the double value
        @Override
        public int compare(Tuple2<String,Double> t1,Tuple2<String,Double> t2) {
            return -(t1._2.compareTo(t2._2)); // sort
        }
    }

    /**
     * parses and returns the provided parameters of execution
     */
    private static class ParamsExtractor {
        private final String[] args;
        private String inputPath;
        private String outputPath;
        private Integer K;
        private Boolean debug;

        public ParamsExtractor(String... args) {
            this.args = args;
        }

        public String getInputPath() {
            return inputPath;
        }

        public String getOutputPath() {
            return outputPath;
        }

        public Boolean getDebug() {
            return debug;
        }

        public Integer getK() {
            return K;
        }

        public ParamsExtractor invoke() {
            if (args.length < 2) {
                System.err.println("Usage: AnomalyIndex <inputfolder> <outputfolder> [K] [DEBUG]");
                System.exit(1);
            }

            // handle input parameters
            inputPath = args[0];
            outputPath = args[1];
           if (args.length > 2  && !args[2].toUpperCase().equals("DEBUG") ) {
                try {
                    K = Integer.parseInt(args[2]);
                } catch (NumberFormatException e) {
                    System.err.println("Argument" + args[2] + " must be an integer.");
                    System.exit(1);
                }
            }
           else K=5;

            debug = (args.length > 2 && args[2].toUpperCase().equals("DEBUG") || args.length > 3 && args[3].toUpperCase().equals("DEBUG"));
            return this;
        }
    }
}
