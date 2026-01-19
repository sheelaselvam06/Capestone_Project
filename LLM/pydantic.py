from pydantic import BaseModel, Field, EmailStr, AnyUrl

class person2(BaseModel):
    name: str
    age: int = 20
    weight: float

r = person2(name="Alice", weight=55.5, age=30)
print(r.name , r.age, r.weight)
# ('Alice', 30, 55.5)


class person(BaseModel):
    name: str = Field(max_length=8)
    age: int = 20
    weight: float
    email: EmailStr
    myurl : AnyUrl

d= {'name':'Bob', 'weight':70.2, 'email':'bob@example.com', 'myurl':'http://example.com'}


r1 = person(**d)
print(r1.name , r1.age, r1.weight, r1.email, r1.myurl)
# Bob 20 70.2 bob@example.com http://example.com/ 