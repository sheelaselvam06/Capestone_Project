from langchain_core.output_parasers import StrOutputparser, JsonOutputparser
 

 str_parser = StrOutputParser()
 print(str_parser.parse("This is a simple string. "))

 json