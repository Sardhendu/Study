package com.hadoop_partitionar;

// This modeule introduces partitionar which uses multiple reducer capability on partition dataset. 
import java.io.IOException;
import java.io.PrintStream;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;

public class HP_mapper
  extends Mapper<LongWritable, Text, IntWritable, IntWritable>
{
  protected void map(LongWritable Key, Text Value, Mapper<LongWritable, Text, IntWritable, IntWritable>.Context context)
  {
    System.out.println("I am  here");
    String[] str_arr = Value.toString().split(" ");
    int[] int_arr = new int[str_arr.length];
    System.out.println("I am here too");
    for (int i = 0; i < str_arr.length; i++) {
      int_arr[i] = Integer.parseInt(str_arr[i]);
    }
    System.out.println("I am here too too");
    System.out.println("The length of int_arr is " + int_arr.length);
    for (int j = 1; j < int_arr.length; j++)
    {
      int i = j - 1;
      int val = int_arr[j];
      while ((i >= 0) && (int_arr[i] > val))
      {
        int_arr[(i + 1)] = int_arr[i];
        i--;
      }
      int_arr[(i + 1)] = val;
      System.out.println("The first value is " + int_arr[(i + 1)]);
    }
    for (int i = 0; i < int_arr.length; i++) {
      System.out.println("look if I am sorted " + int_arr[i]);
    }
    System.out.println("I am here too too too");
    for (int i = 0; i < int_arr.length; i++)
    {
      System.out.println("The first value is mmm " + int_arr[i]);
      try
      {
        context.write(new IntWritable(int_arr[i]), new IntWritable(1));
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
}
