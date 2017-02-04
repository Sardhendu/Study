package com.hadoop_comparable;

import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class HC_driver
{
  public static void main(String[] args)
    throws IOException, InterruptedException, ClassNotFoundException
  {
    Job job_run = new Job(new Configuration());
    job_run.setJobName("I dont know what this is");
    
    job_run.setJarByClass(HC_driver.class);
    job_run.setMapperClass(HC_mapper.class);
    job_run.setCombinerClass(HC_reducer.class);
    job_run.setReducerClass(HC_reducer.class);
    
    job_run.setMapOutputKeyClass(Text.class);
    job_run.setMapOutputValueClass(HC_comparable.class);
    job_run.setOutputKeyClass(Text.class);
    job_run.setOutputValueClass(HC_comparable.class);
    
    FileInputFormat.setInputPaths(job_run, new Path[] { new Path("/home/hduser/in_stud_details.txt") });
    FileOutputFormat.setOutputPath(job_run, new Path("/home/hduser/Writable_output"));
    
    job_run.waitForCompletion(true);
    
    String val = job_run.getConfiguration().get("prop-name");
  }
}
