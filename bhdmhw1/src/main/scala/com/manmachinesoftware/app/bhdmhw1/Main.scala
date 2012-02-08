package com.manmachinesoftware.app.bhdmhw1

import java.util.ArrayList
import java.io.File
import io.Source
import collection.immutable
import collection.mutable
import collection.mutable.{HashMap,ListBuffer}
import collection.immutable.{Map => imMap}
import scalala.tensor.counters.Counters.DefaultIntCounter
import scalanlp.data.{LabeledDocument}
import scalala.tensor.sparse.SparseVector
import util.Random


object Main
{
    val NEG = -1
    val POS = 1
    val featureBody = "reviewBody"
    val google = new java.io.File(getClass().getResource("/stopwords/googleStop.txt").getFile())
    val mysql = new java.io.File(getClass().getResource("/stopwords/mysqlStop.txt").getFile())
    val ranknlAll = new java.io.File(getClass().getResource("/stopwords/ranknlAllStop.txt").getFile())
    val ranknl = new java.io.File(getClass().getResource("/stopwords/ranknlStop.txt").getFile())
    var xFold = 10

    //Parameters
    val useStopWords = google

    var stopWords = new Array[String](0)
    if (useStopWords != null) {
        val stopWordsSource = Source.fromFile(useStopWords)
        stopWords = stopWordsSource.mkString.split("\\s+")
        stopWordsSource.close()
    }

    // Test conditions (bernoulli),stem,stop,smoothing = 8Xcont different conditions
    def main(args: Array[String]) = 
    {
        var corpus = Random.shuffle(getData())

        //Bernoulli bayes without stemming or stopwords
        var (corpOrig,dictOrig,binDictOrig,idxToWOrig,wToIdxOrig) = mapData(corpus.toList,stopWords,true)
        //Generate sets from the already Ramdomly ordered corpus
        val sets = corpOrig.sliding(corpus.size/xFold)

        //For each set train on everything else, but that set
        for (t <- 0 until sets.length) {
            //Train against the tail


        
            //Run the query on the head
            query()//Vec for data and two vecs one for each class?
            f1score()
        }

        //Return results for each query

        //Compute F1 scores
        //Plot F1 scores
        //Print distinguishing words


        //Multinomial Bayes with stemming
    }

    def runBayes(data:Seq[Seq[LabeledDocument[Double,String]]],alpha:Double = 1,bin:Boolean = true) = 
    {
        //Convert to vectors
        //Do bernouli or not
        //Do bayes
        //Run on data set
        //Return relabeled documents,two vocabulary vectors
        
    }

    def f1score() = 
    {
        //Given set of pos,neg and actual set
        //Count real by ID and then return score on query
    }

    def mapData(corp:List[LabeledDocument[Double,String]], sW:Array[String],useStemmer:Boolean ):
    (List[LabeledDocument[Double,String]],HashMap[String,Int],HashMap[String,Int],Seq[(String,Int)],HashMap[String,Int])= 
    {

        var corpus = corp
        var stopWords = sW
        //Map stem and stopwords onto corpus
        if (useStemmer) {
            corpus = corpus.map( (doc) => new LabeledDocument[Double,String](doc.id,doc.label,stemDoc(doc.fields)))
            stopWords.map(stemmerRun.porterStem)
        }
        if (stopWords.size != 0) {
            corpus = corpus.map( (doc) => new LabeledDocument[Double,String](doc.id,doc.label,stopDoc(doc.fields,stopWords)))
        }

        
        //Build dictionary and index
        var dict = new HashMap[String,Int]()
        var binDict = new HashMap[String,Int]()
        for (doc:LabeledDocument[Double,String] <- corpus.toIndexedSeq) {
            for ( i <- 0 until (doc.fields.get(featureBody).size)) {
                val ss = doc.fields.get(featureBody).get(i)
                dict.put(ss,dict.getOrElse(ss,0)+1)
                binDict.put(ss,1)
            }
        }
        
        //Create mapping to/from idx to w
        var idxToW = dict.toIndexedSeq
        var wToIdx = new HashMap[String,Int]()
        var i = 0
        for (x <- idxToW) {
            wToIdx.put(x._1,i)
            i = i + 1
        }
        return (corpus,dict,binDict,idxToW,wToIdx)
    }
    
    def count(doc:LabeledDocument[Double,String]):mutable.Map[String,Int] = 
    {   
        val words:Seq[String] = doc.fields.getOrElse(featureBody,null)
        val dict = mutable.Map[String,Int]()
        words.map((s:String) => dict.put(s,dict.getOrElse[Int](s,0) +1))
        return dict
    }

    def addDict(a:mutable.Map[String,Int],b:mutable.Map[String,Int]):mutable.Map[String,Int] = {
        var c = new HashMap[String,Int]()
        for ( k <- a.keys ) {
            c.put(k,a.getOrElse(k,0)) 
        }
        for ( k <- b.keys ) {
            c.put(k,b.getOrElse(k,0)) 
        }
        return c
    }

    def getData():ListBuffer[LabeledDocument[Double,String]] = 
    {
        var corpus = new ListBuffer[LabeledDocument[Double,String]]()
        //Open resource directories for +ve and -ve
        val fnegPath = new java.io.File(getClass().getResource("/polarityData/neg").getFile())
        val fposPath = new java.io.File(getClass().getResource("/polarityData/pos").getFile())
        val fneg = fnegPath.list()
        val fpos = fposPath.list()
        
        //Build corpus
        for (review <- fneg) {
            val seq = parseFile(new File(fnegPath,review))
            corpus += (new LabeledDocument(review,NEG,immutable.HashMap((featureBody,seq))))
        }
        for (review <- fpos) {
            val seq = parseFile(new File(fposPath,review))
            corpus += (new LabeledDocument(review,POS,immutable.HashMap((featureBody,seq))))
        }
        return corpus
    }

    def parseFile(f:File):Seq[String] = 
    {
        val file = Source.fromFile(f)
        var string = file.mkString.toLowerCase
        file.close()
        
        string = string.replaceAll("(?<=\\w)-\\s*\n\\s*","") //From SimpleEnglishParser
        string = string.replaceAll("n't", " not")
        string = string.replaceAll("in'", "ing")
        string = string.replaceAll("'re", " are")
        string = string.replaceAll("'ve", " have")
        string = string.replaceAll("'--+"," ") 
        string = string.replaceAll("'s|[\\p{Punct}^-]"," ") //remove possesives and punctuation
        string = string.replaceAll("\\s+","\n")
        return string.split("\\s+").map((s:String) => s.trim)
    }

    def stemDoc( fields:imMap[String,Seq[String]]): imMap[String,Seq[String]] = 
    {
        var dict:HashMap[String,Seq[String]] = HashMap[String,Seq[String]]()
        var words = ListBuffer[String]()
        var terms:Seq[String] = fields.getOrElse(featureBody,null) 
        for ( i <- 0 until terms.size) {
            var ss = terms(i)
            words.append(stemmerRun.porterStem(ss))
        }
        dict.put(featureBody,words)
        return dict.toMap
    }

    def stopDoc( fields:imMap[String,Seq[String]],stopWords:Array[String]): imMap[String,Seq[String]] = 
    {
        var dict:HashMap[String,Seq[String]] = HashMap[String,Seq[String]]()
        var words = ListBuffer[String]()
        var terms:Seq[String] = fields.getOrElse(featureBody,null) 
        for ( i <- 0 until terms.size) {
            val ss = terms(i)
            if (stopWords.contains(ss) == false) { words.append(ss) } 
        }
        dict.put(featureBody,words)
        return dict.toMap
    }

    def encode(dict:Map[String,Double],wToIdx:Map[String,Int],bit: Boolean = false):SparseVector = 
    {
        var vec = new SparseVector(dict.size)
        for (x:(String,Double) <- dict.toIndexedSeq) {
            var count = x._2
            if (bit && count>0) {
                count = 1
            }   
            vec.update(wToIdx.get(x._1),count)    
        }
        return vec
    }

    def decode(vec:SparseVector,idxToW:IndexedSeq[(String,Double)]):HashMap[String,Double] = 
    {
        var dict = new HashMap[String,Double]()
        for (k <- vec.activeKeys) {
            dict.put(idxToW(k)._1,vec(k))
        }
        return dict
    }
}

// vim: set ts=4 sw=4 et:
