mkdir -p pics

for i in \
	"example 6.1" \
	"example 6.2" \
	"example 6.3" \
	"example 6.3 without convexity";
	do \
		date > "$i.log"

		# Run the example
		{ time python3 projected_gradient_method.py "$i" ; } >> "$i.log" 2>&1

		# Extract nice pictures
		python export_pics.py "$i" >> "$i.log" 2>&1 

		# Remove the solution files
		rm -rf "solutions/$i/"
done
