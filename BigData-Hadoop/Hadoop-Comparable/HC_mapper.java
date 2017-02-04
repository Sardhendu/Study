
package com.hadoop_comparable;

import java.io.IOException;
import java.io.PrintStream;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;

public class HC_mapper
  extends Mapper<LongWritable, Text, Text, HC_comparable>
{
  private HC_comparable outAverage = new HC_comparable();
  private Text id = new Text();
  
  public void map(LongWritable key, Text line, Mapper<LongWritable, Text, Text, HC_comparable>.Context context)
    throws InterruptedException, IOException
  {
    System.out.println("Inside mapper class");
    String[] tokens = line.toString().split(",");
    if (tokens.length < 10)
    {
      System.out.println("I am inside token.length if stat " + tokens[1]);
      return;
    }
    System.out.println("The tokens are as follow");
    for (int i = 0; i < tokens.length; i++) {
      System.out.println(tokens[i]);
    }
    String dateTime = tokens[0];
    String stationId = tokens[1];
    String trafficCount = tokens[9];
    System.out.println("The traffic count array is " + tokens[9]);
    if (trafficCount.length() > 0)
    {
      this.id.set(stationId + "_" + dateTime);
      System.out.println("Just pagalpanti");
      this.outAverage.set(1, Integer.parseInt(trafficCount));
      
      context.write(this.id, this.outAverage);
    }
  }
}
