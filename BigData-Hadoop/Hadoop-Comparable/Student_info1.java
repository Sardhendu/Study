package com.hadoop_comparable;

import java.io.PrintStream;

class student_info_1
{
  private String stud_name;
  private int stud_id;
  
  public student_info_1(int stud_id, String stud_name)
  {
    System.out.println("The id and name are inserted into the list");
    this.stud_id = stud_id;
    this.stud_name = stud_name;
  }
  
  public int get_id()
  {
    return this.stud_id;
  }
  
  public void set_id(int stud_id)
  {
    this.stud_id = stud_id;
  }
  
  public String get_name()
  {
    return this.stud_name;
  }
  
  public void set_name(String stud_name)
  {
    this.stud_name = stud_name;
  }
  
  public String toString()
  {
    System.out.println("I am here in toString");
    
    return this.stud_id + ": " + this.stud_name;
  }
}
