package com.manmachinesoftware.app.bhdmhw1

object Main
{
    def main(args: Array[String]) = 
    {
        //Open resource directories for +ve and -ve

        val fneg = new java.io.File(getClass().getResource("/polarityData/neg").getFile()).list()
        val fpos = new java.io.File(getClass().getResource("/polarityData/pos").getFile()).list()

        for ( x <- fneg ) {
            println(x)
        }

    }

}

// vim: set ts=4 sw=4 et:
