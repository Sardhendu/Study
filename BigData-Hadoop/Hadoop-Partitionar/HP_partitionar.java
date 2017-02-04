package com.hadoop_partitionar;

import java.io.PrintStream;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Partitioner;

public class HP_partitionar
  extends Partitioner<IntWritable, IntWritable>
{
  public int getPartition(IntWritable Key, IntWritable Value, int num_of_part)
  {
    int curr_val = Key.get();
    
    System.out.println("The key in Partitionar is " + curr_val);
    System.out.println("BaluBaluBaluBalu");
    if (curr_val <= 20) {
      return 0;
    }
    return 1;
  }
}
