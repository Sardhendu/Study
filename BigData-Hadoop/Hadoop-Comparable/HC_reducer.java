package com.hadoop_comparable;

import java.io.IOException;
import java.io.PrintStream;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;

public class HC_reducer
  extends Reducer<Text, HC_comparable, Text, HC_comparable>
{
  private HC_comparable outAverage = new HC_comparable();
  
  public void reduce(Text key, Iterable<HC_comparable> averages, Reducer<Text, HC_comparable, Text, HC_comparable>.Context context)
    throws InterruptedException, IOException
  {
    System.out.println("Inside reducer class");
    double sum = 0.0D;
    int numElements = 0;
    for (HC_comparable partialAverage : averages)
    {
      System.out.println("partialaverage getnum is " + partialAverage.getNumElements());
      System.out.println("partialaverage is " + partialAverage.getAverage());
      
      sum += partialAverage.getAverage() * partialAverage.getNumElements();
      numElements += partialAverage.getNumElements();
    }
    double average = sum / numElements;
    System.out.println("the final average is " + average);
    
    this.outAverage.set(numElements, average);
    System.out.println("OutAverage is " + this.outAverage.getAverage());
    context.write(key, this.outAverage);
  }
}
