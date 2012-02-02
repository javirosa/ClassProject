package com.manmachinesoftware.app.bhdmhw1

import java.util.ArrayList
import java.io.File
import scala.io.Source
import scala.collection.immutable
import scala.collection.mutable.HashMap
import scalala.tensor.counters.Counters.DefaultIntCounter
import scalanlp.data.{LabeledDocument}
import scalanlp.data.{Example}


object Main
{
    val NEG = -1
    val POS = 1

    def main(args: Array[String]) = {
       getData()
    }

    //TODO
    // Do I need to build a master dictionary?
    // What do I give the LabeledDocument to get it to work with the Bayes algorithm
    // -- what is the argument to the Labeled Document constructor?
    // Assuming Naive Bayes works how do I compute F1 Scores
    // How do I determine what the interesting keywords are?
    // How do I plot the word counts and F1 score performance measures
    // How do I generate subsets of data?
    def getData() = {

        //Open resource directories for +ve and -ve
        val fnegPath = new java.io.File(getClass().getResource("/polarityData/neg").getFile())
        val fposPath = new java.io.File(getClass().getResource("/polarityData/pos").getFile())
        val fneg = fnegPath.list()
        val fpos = fposPath.list()

        val google = new java.io.File(getClass().getResource("/stopwords/googleStop.txt").getFile())
        val mysql = new java.io.File(getClass().getResource("/stopwords/mysqlStop.txt").getFile())
        val ranknlAll = new java.io.File(getClass().getResource("/stopwords/ranknlAllStop.txt").getFile())
        val ranknl = new java.io.File(getClass().getResource("/stopwords/ranknlStop.txt").getFile())
        
        //Build corpus
        var corpus = new ArrayList[LabeledDocument[Double,String]](fneg.length + fpos.length)
        for (review <- fneg) {
            val seq = parseFile(new File(fnegPath,review),stopWords = google, stem=true)
            corpus.add(new LabeledDocument(review,0,immutable.HashMap(("review",seq))))
        }
        println(corpus.toString)
    }

    def parseFile(f:File,stopWords:File = null,stem:Boolean = false):Seq[String] = {
        val file = Source.fromFile(f)
        var string = file.mkString
        file.close()
        string = string.replaceAll("\\s+","\n").replaceAll("'s|'|\\p{Punct}","") //remove prossesives and punctuation
        
        var text = string.split("\\s+").map((s:String) => s.trim.toLowerCase)
        if (stopWords != null) {
            val stopWord= Source.fromFile(f)
            val words = stopWord.mkString.split("\\s+")
            text = text.filterNot((s:String) => words.contains(s))
        }
        if (stem) {
           text = text.map(porterStem)
        }
        return text.toIndexedSeq //ArrayBuffer
    }

    def porterStem(s:String):String = { 
        var stemmer = new Stemmer()
        stemmer.add(s.trim())
        if (stemmer.b.length >2 ) {
            stemmer.step1
            stemmer.step2
            stemmer.step3
            stemmer.step4
            stemmer.step5a
            stemmer.step5b
        }
        return stemmer.b
    }


}

// vim: set ts=4 sw=4 et:
