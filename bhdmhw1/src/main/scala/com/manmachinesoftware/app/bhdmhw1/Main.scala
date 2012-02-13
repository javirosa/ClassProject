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
    val LABELS = List(NEG,POS)
    val featureBody = "reviewBody"
    val google = new java.io.File(getClass().getResource("/stopwords/googleStop.txt").getFile())
    val mysql = new java.io.File(getClass().getResource("/stopwords/mysqlStop.txt").getFile())
    val ranknlAll = new java.io.File(getClass().getResource("/stopwords/ranknlAllStop.txt").getFile())
    val ranknl = new java.io.File(getClass().getResource("/stopwords/ranknlStop.txt").getFile())


    var swGoogle = readSW(google).toArray
    var swMysql =  readSW(mysql).toArray
    var swRanknlAll = readSW(ranknlAll).toArray
    var swRanknl = readSW(ranknl).toArray
    var swNone = Array[String]()


    //TEST CONDITIONS
    var xFold = 10
    util.Random.setSeed(0)
    val swords = swRanknlAll
    val stem = true
    val bernoulli = true
    val alphaSmoothing = 1


    def main(args: Array[String]) = 
    {
        //Randomize the data inputs for later separation into query and training sets
        var corpus = Random.shuffle(getData())

        //Bernoulli bayes without stemming or stopwords
        var (corpOrig,dictOrig,binDictOrig,idxToWOrig,wToIdxOrig) = mapData(corpus.toList,swords,stem)
        //Generate sets from the already Ramdomly ordered corpus
        val sets = corpOrig.grouped(corpOrig.size/xFold).toSeq

        //For each set train on everything else, but that set
        val idxSize = wToIdxOrig.size
        for (i <- 0 until sets.size)
        {
            val train = (sets.splitAt(i)._1 ++ sets.splitAt(i+1)._2).flatten
            val classCorpus = train.groupBy((doc) => doc.label)


            //Map documents to vectors
            val booleanVecs = classCorpus.mapValues(
            (seq) => seq.map(
                (doc) => encode(wToIdxOrig.toMap,bernoulli,count(doc).toMap)
                )
            )
            var termClassProbs = HashMap[Int,SparseVector](NEG -> new SparseVector(idxSize), POS -> new SparseVector(idxSize))
            termClassProbs.foreach( 
                (kv) => { 
                    val lbl = kv._1; val vec = kv._2;
                    val classSize = booleanVecs.getOrElse(lbl,null).size
                    val prior = classSize.toDouble/(train.size)
                    for (i <- 0 until idxSize) vec.update(i, 0)
                    booleanVecs.getOrElse(lbl,null).foreach( (v) => vec += v)
                    val totOccurances:Double = if(bernoulli) classSize.toDouble else vec.reduceLeft[(Int,Double)]((x,b) => (0,x._2 + b._2))._2
                    for (i <- 0 until idxSize) vec.update(i, math.log(((alphaSmoothing+vec(i))*prior)/(totOccurances+alphaSmoothing*idxSize))) 
                }
            )

            //For each vector in query apply to each class vector 
            val query = sets(i)
            val qClass = query.groupBy( doc => doc.label)
            var qClassVecs = qClass.mapValues(
                seq => seq.map(count).map(
                    x => {
                        //Classify
                        val vec = encode(wToIdxOrig.toMap,bernoulli,x.toMap)
                        val pvec = termClassProbs.getOrElse(POS,null).dot(vec)
                        val nvec = termClassProbs.getOrElse(NEG,null).dot(vec)
                        if (pvec >= nvec) POS else NEG
                    }
                ).groupBy(x => x)
            )

            //Compute F1 scores
            val tp = qClassVecs.getOrElse(POS,null).getOrElse(POS,Seq()).size 
            val tn = qClassVecs.getOrElse(NEG,null).getOrElse(NEG,Seq()).size 
            val fn = qClassVecs.getOrElse(POS,null).getOrElse(NEG,Seq()).size 
            val fp = qClassVecs.getOrElse(NEG,null).getOrElse(POS,Seq()).size 

            println("#f1:%f tp:%d fn:%d fp:%d tn:%d".format(f1score(tp,fn,fp),tp,fn,fp,tn))
            println("%f,%d,%d,%d,%d".format(f1score(tp,fn,fp),tp,fn,fp,tn))

            //Print distinguishing words
            for (x <-LABELS) {
                println("Class:%d".format(x))
                decode(idxToWOrig,termClassProbs.getOrElse(x,null)).toList.sortWith((x,y) => x._2 >= y._2).take(10).
                            foreach(t => print("%s:%2.3f ".format(t._1,t._2)))
                println();
            }
        }
    }

    def f1score(tp:Int,fn:Int,fp:Int):Double = 
    {
        //Get precision and recall:  tp/(tp+fp), tp/(tp + fn)
        val recall = tp.toDouble/(tp+fn)
        val relevance = tp.toDouble/(tp+fp)
        return 2*recall*relevance/(recall + relevance)
    }

    def mapData(corp:List[LabeledDocument[Double,String]], sW:Array[String],useStemmer:Boolean ):
    (List[LabeledDocument[Double,String]],HashMap[String,Double],HashMap[String,Int],IndexedSeq[(String,Double)],HashMap[String,Int]) = 
    {

        var corpus = corp
        var stopWords = sW
        //Map stem and stopwords onto corpus
        if (useStemmer) {
            corpus = corpus.map( (doc) => new LabeledDocument[Double,String](doc.id,doc.label,stemDoc(doc.fields)))
            stopWords = stopWords.map(stemmerRun.porterStem)
        }
        if (stopWords.size != 0) {
            corpus = corpus.map( (doc) => new LabeledDocument[Double,String](doc.id,doc.label,stopDoc(doc.fields,stopWords)))
        }

        
        //Build dictionary and index
        var dict = new HashMap[String,Double]()
        var binDict = new HashMap[String,Int]()
        for (doc:LabeledDocument[Double,String] <- corpus.toIndexedSeq) {
            for ( i <- 0 until (doc.fields.get(featureBody).size)) {
                val ss = doc.fields.get(featureBody).get(i)
                dict.put(ss,dict.getOrElse(ss,0.0)+1.0)
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

    def encode(wToIdx:Map[String,Int],bit: Boolean = false,docDict:Map[String,Int]):SparseVector = 
    {
        var vec = new SparseVector(wToIdx.size)
        for (i <- 0 until vec.size) vec.update(i,0)
        for (x:(String,Int) <- docDict.toIndexedSeq) {
            var count = x._2
            if (bit && count>0) {
                count = 1
            }   
            vec.update(wToIdx.get(x._1),count)    
        }
        return vec
    }

    def decode(idxToW:IndexedSeq[(String,Double)],vec:SparseVector):HashMap[String,Double] = 
    {
        var dict = new HashMap[String,Double]()
        for (k <- vec.activeKeys) {
            dict.put(idxToW(k)._1,vec(k))
        }
        return dict
    }

    def readSW(sourceFile:java.io.File):Seq[String] = {
        val stopWordsSource = Source.fromFile(sourceFile)
        val stopWords = stopWordsSource.mkString.split("\\s+")
        stopWordsSource.close()
        return stopWords
    }

}

// vim: set ts=4 sw=4 et:
