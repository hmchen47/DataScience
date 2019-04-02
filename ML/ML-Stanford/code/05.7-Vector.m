prediction = 0.0;
for j = 1:n+1,
  prediction += theta(j) * x(j);
end;


prediction = theta' * x;


