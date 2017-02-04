package com.hadoop_comparable;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class BBBBB
{
  public void addelements(ArrayList<student_info_1> studnameid_list)
  {
    System.out.println("before 1st row");
    studnameid_list.add(new student_info_1(1, "sam"));
    System.out.println("After 1st row");
    studnameid_list.add(new student_info_1(7, "vam"));
    studnameid_list.add(new student_info_1(6, "mam"));
    studnameid_list.add(new student_info_1(2, "jam"));
    studnameid_list.add(new student_info_1(3, "lam"));
    studnameid_list.add(new student_info_1(5, "pam"));
    studnameid_list.add(new student_info_1(4, "wam"));
    studnameid_list.add(new student_info_1(8, "jam"));
    System.out.println("The add of collection is done");
  }
  
  public void dispelements(ArrayList<student_info_1> studnameid_list)
  {
    for (student_info_1 stud : studnameid_list)
    {
      System.out.println("I am inside display");
      System.out.println(stud);
    }
  }
  
  public static void main(String[] args)
  {
    List<student_info_1> studnameid_list = new ArrayList();
    
    BBBBB stud_list = new BBBBB();
    stud_list.addelements((ArrayList)studnameid_list);
    System.out.println("I am in the transition of add and display");
    stud_list.dispelements((ArrayList)studnameid_list);
    
    Collections.sort(studnameid_list, new Comparator()
    {
      public int compare(student_info_1 s1, student_info_1 s2)
      {
        if (s1.get_id() > s2.get_id()) {
          return 1;
        }
        if (s1.get_id() < s2.get_id()) {
          return -1;
        }
        return 0;
      }
    });
    stud_list.dispelements((ArrayList)studnameid_list);
    
    Collections.sort(studnameid_list, new Comparator()
    {
      public int compare(student_info_1 s1, student_info_1 s2)
      {
        return s1.get_name().compareTo(s2.get_name());
      }
    });
    stud_list.dispelements((ArrayList)studnameid_list);
  }
}
