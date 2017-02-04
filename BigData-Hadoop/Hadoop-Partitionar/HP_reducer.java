package com.hadoop_partitionar;

import java.io.IOException;
import java.io.PrintStream;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;

public class HP_reducer
  extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable>
{
  public void reduce(IntWritable Key, Iterable<IntWritable> value_list, Reducer<IntWritable, IntWritable, IntWritable, IntWritable>.Context context)
  {
    System.out.println("The key is " + Key);
    int sum = 0;
    for (IntWritable i : value_list)
    {
      System.out.println("The value is " + i.get());
      sum += i.get();
    }
    try
    {
      context.write(Key, new IntWritable(sum));
    }
    catch (IOException e)
    {
      e.printStackTrace();
    }
    catch (InterruptedException e)
    {
      e.printStackTrace();
    }
  }
}
