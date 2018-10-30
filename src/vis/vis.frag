R"( /*"*/

out vec4 fragcolor;

uniform int mode;

void main()
{
	switch (mode)
	{
	case 0: fragcolor = vec4(.5, .5, .5, 1.); break;		// line mode
	case 1: fragcolor = vec4(.6, .6, 1., 1.); break;		// fragment mode
	}
}

/*"*/)"